import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple, Union
import math
from diffusers import StableDiffusionPipeline
import gc
import os
# override unet forward
# The only difference from diffusers:
# return intermediate UNet features of all UpSample blocks
def override_forward(self):

    def forward(
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        added_cond_kwargs: Optional[Dict[str, Any]] = None,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
        last_up_block_idx: int = None,
    ):
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        t = timestep
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

                # `Timesteps` does not contain any weights and will always return f32 tensors
                # there might be better ways to encapsulate this.
                class_labels = class_labels.to(dtype=sample.dtype)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)

            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        if self.config.addition_embed_type == "text":
            aug_emb = self.add_embedding(encoder_hidden_states)
            emb = emb + aug_emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        if self.encoder_hid_proj is not None:
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down 
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples += (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual

        # 5. up
        # only difference from diffusers:
        # save the intermediate features of unet upsample blocks
        # the 0-th element is the mid-block output
        if return_intermediates:
            all_intermediate_features = [sample]
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )
            if return_intermediates:
                all_intermediate_features.append(sample)

            if last_up_block_idx is not None and i == last_up_block_idx and return_intermediates:
                return all_intermediate_features

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        # only difference from diffusers, return intermediate results
        if return_intermediates:
            return sample, all_intermediate_features
        else:
            return sample

    return forward


class MovePipeline(StableDiffusionPipeline):

    # must call this function when initialize
    def modify_unet_forward(self):
        self.unet.forward = override_forward(self.unet)

    def inv_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta=0.,
        verbose=False
    ):
        """
        Inverse sampling for DDIM Inversion
        """
        if verbose:
            print("timestep: ", timestep)
        next_step = timestep
        timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999)
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_step]
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_next)**0.5 * model_output
        x_next = alpha_prod_t_next**0.5 * pred_x0 + pred_dir
        return x_next, pred_x0
    
    def pred_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        pred_x0:torch.FloatTensor
    ):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep > 0 else self.scheduler.final_alpha_cumprod
        pred_dir = (1 - alpha_prod_t_prev)**0.5 * model_output
        x_prev = alpha_prod_t_prev**0.5 * pred_x0 + pred_dir
        return x_prev
    
    def get_interxt(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        pred_x0:torch.FloatTensor
    ):
        # next_step = timestep
        # timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999)
        # alpha_prod_t_next = self.scheduler.alphas_cumprod[next_step]
        # pred_dir = (1 - alpha_prod_t_next)**0.5 * model_output
        # x_next = alpha_prod_t_next**0.5 * pred_x0[2:3] + pred_dir

        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep > 0 else self.scheduler.final_alpha_cumprod
        x_t = alpha_prod_t**0.5 * pred_x0 + (1 - alpha_prod_t)**0.5 * model_output
        return x_t

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
    ):
        """
        predict the sample of the next step in the denoise process.
        """
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep > 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_prev)**0.5 * model_output
        x_prev = alpha_prod_t_prev**0.5 * pred_x0 + pred_dir
        return x_prev, pred_x0

    @torch.no_grad()
    def image2latent(self, image):
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if type(image) is Image:
            image = np.array(image)
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        # input image density range [-1, 1]
        latents = self.vae.encode(image)['latent_dist'].mean
        latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        elif return_type == "pt":
            image = (image / 2 + 0.5).clamp(0, 1)

        return image

    def latent2image_grad(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents)['sample']

        return image  # range [-1, 1]

    @torch.no_grad()
    def get_text_embeddings(self, prompt):
        # text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_input.input_ids.cuda())[0]
        return text_embeddings

    # get all intermediate features and then do bilinear interpolation
    # return features in the layer_idx list
    def forward_unet_features(
            self, 
            z, 
            t, 
            encoder_hidden_states, 
            guidance_3d,
            guidance_traj,
            layer_idx=[0]
        ):
        
        if guidance_3d is not None and guidance_traj is not None:
            use_guidance = True
        else:
            use_guidance = False

        guidance_3d = guidance_3d.repeat(encoder_hidden_states.shape[0], 1, 1)
        guidance_traj = guidance_traj.repeat(encoder_hidden_states.shape[0], 1, 1)

        encoder_hidden_states = torch.cat([encoder_hidden_states, guidance_3d, guidance_traj], dim=1)

        unet_output, all_intermediate_features = self.unet(
            z,
            t,
            encoder_hidden_states=encoder_hidden_states,
            return_intermediates=True
            )

        all_return_features = {}

        for idx in layer_idx:
            all_return_features[idx] = all_intermediate_features[idx]

        return unet_output, all_return_features
    
    def forward_unet_features_guide_cond(
            self, 
            z, 
            t, 
            encoder_hidden_states, 
            guidance_3d,
            guidance_traj,
            layer_idx=[0]
        ):
        
        # added_cond_kwargs = {
        #     "guidance_3d": guidance_3d,
        #     "guidance_traj": guidance_traj
        # }
        
        unet_output, all_intermediate_features = self.unet(
            z,
            t,
            encoder_hidden_states=encoder_hidden_states,
            # added_cond_kwargs=added_cond_kwargs,
            return_intermediates=True
            )

        all_return_features = {}

        for idx in layer_idx:
            all_return_features[idx] = all_intermediate_features[idx]

        return unet_output, all_return_features

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        prompt_embeds=None, # whether text embedding is directly provided.
        guidance_3d=None,
        guidance_traj=None,
        batch_size=1,
        height=512,
        width=512,
        num_inference_steps=50,
        num_actual_inference_steps=None,
        guidance_scale=7.5,
        latents=None,
        pred_x0=None,
        unconditioning=None,
        neg_prompt=None,
        save_dir=None,
        return_intermediates=False,
        **kwds):
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if guidance_3d is not None and guidance_traj is not None:
            use_guidance = True
        else:
            use_guidance = False

        if guidance_scale > 1.:
            guidance_scale = 1.

        if prompt_embeds is None:
            if isinstance(prompt, list):
                batch_size = len(prompt)
            elif isinstance(prompt, str):
                if batch_size > 1:
                    prompt = [prompt] * batch_size

            # text embeddings
            text_input = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        else:
            batch_size = prompt_embeds.shape[0]
            text_embeddings = prompt_embeds
        print("prompt: ", prompt)
        print("input text embeddings :", text_embeddings.shape)

        # add guidance to text embeddings
        if use_guidance:
            if batch_size > 1:
                guidance_3d = guidance_3d.repeat(batch_size, 1, 1)
                guidance_traj = guidance_traj.repeat(batch_size, 1, 1)
            text_embeddings = torch.cat([text_embeddings, guidance_3d, guidance_traj], dim=1)
            print("text embeddings with guidance: ", text_embeddings.shape)

        # define initial latents if not predefined
        if latents is None:
            latents_shape = (batch_size, self.unet.in_channels, height//8, width//8)
            latents = torch.randn(latents_shape, device=DEVICE, dtype=self.vae.dtype)

        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.:
            if neg_prompt:
                uc_text = neg_prompt
            else:
                uc_text = ""
            unconditional_input = self.tokenizer(
                [uc_text] * batch_size,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]

            if use_guidance:
                guidance_3d_length = guidance_3d.shape[-2]
                unconditional_input_3d = np.zeros((guidance_3d_length * 8, 3))
                unconditional_input_3d_tensor = torch.tensor(unconditional_input_3d, device=DEVICE, dtype=torch.float32)
                unconditional_embeddings_3d = self.spatial_guidance_model(unconditional_input_3d_tensor)
                unconditional_embeddings_3d = unconditional_embeddings_3d.repeat(batch_size, 1, 1)

                guidance_traj_length = guidance_traj.shape[-2]
                unconditional_input_traj = np.zeros((guidance_traj_length * 2, 7))
                unconditional_input_traj_tensor = torch.tensor(unconditional_input_traj, device=DEVICE, dtype=torch.float32)
                unconditional_embeddings_traj = self.trajectory_guidance_model(unconditional_input_traj_tensor)
                unconditional_embeddings_traj = unconditional_embeddings_traj.repeat(batch_size, 1, 1)

                unconditional_embeddings = torch.cat([unconditional_embeddings, unconditional_embeddings_3d, unconditional_embeddings_traj], dim=1)

            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)
            print("cfg text embeddings shape: ", text_embeddings.shape)

        print("latents shape: ", latents.shape)
        # iterative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        # print("Valid timesteps: ", reversed(self.scheduler.timesteps))
        if return_intermediates:
            latents_list = [latents]
            noise_pred_list= []
        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="DDIM Sampler")):
            if num_actual_inference_steps is not None and i < num_inference_steps - num_actual_inference_steps:
                continue

            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents

            if unconditioning is not None and isinstance(unconditioning, list):
                _, text_embeddings = text_embeddings.chunk(2)
                text_embeddings = torch.cat([unconditioning[i].expand(*text_embeddings.shape), text_embeddings]) 
            # predict the noise

            # compute the previous noise sample x_t -> x_t-1
            # YUJUN: right now, the only difference between step here and step in scheduler
            # is that scheduler version would clamp pred_x0 between [-1,1]
            # # don't know if that's gonna have huge impact

            # added_cond_kwargs = {
            #     "guidance_3d": guidance_3d,
            #     "guidance_traj": guidance_traj
            # }

            if i == num_inference_steps-num_actual_inference_steps and pred_x0 != None:
                noise_pred = self.unet(
                    model_inputs, 
                    t, 
                    encoder_hidden_states=text_embeddings
                    # added_cond_kwargs=added_cond_kwargs
                    )
                inter_xt = self.get_interxt(noise_pred[1:-1], t, pred_x0[1:-1])
                latents[1:-1] = inter_xt
                noise_pred = self.unet(
                    latents, 
                    t, 
                    encoder_hidden_states=text_embeddings
                    # added_cond_kwargs=added_cond_kwargs
                    )
                latents = self.pred_step(noise_pred, t, pred_x0) 
            else:
                noise_pred = self.unet(
                    model_inputs, 
                    t, 
                    encoder_hidden_states=text_embeddings,
                    # added_cond_kwargs=added_cond_kwargs
                    )
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if return_intermediates:
                latents_list.append(latents)
                noise_pred_list.append(noise_pred)

        if latents.shape[0]>6:
            image = []
            for i in range(math.ceil((latents.shape[0])/6)):
                gc.collect()
                torch.cuda.empty_cache()
                image.append(self.latent2image(latents[6*i:6*(i+1)], return_type="pt"))
            image = torch.cat(image,dim=0)
        else: image = self.latent2image(latents, return_type="pt")
        if return_intermediates:
            return image, latents_list
        return image

    @torch.no_grad()
    def invert(
        self,
        image: torch.Tensor,
        prompt,
        guidance_3d,
        guidance_traj,
        num_inference_steps=50,
        num_actual_inference_steps=None,
        guidance_scale=7.5,
        eta=0.0,
        return_intermediates=False,
        **kwds):
        """
        invert a real image into noise map with determinisc DDIM inversion
        """
        if guidance_3d is not None and guidance_traj is not None:
            use_guidance = True
        else:
            use_guidance = False

        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        batch_size = image.shape[0]
        if isinstance(prompt, list):
            if batch_size == 1:
                image = image.expand(len(prompt), -1, -1, -1)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        if use_guidance and batch_size > 1:
            guidance_3d = guidance_3d.repeat(batch_size, 1, 1)
            guidance_traj = guidance_traj.repeat(batch_size, 1, 1)

        # text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        print("input text embeddings :", text_embeddings.shape)

        # add guidance to text embeddings
        if use_guidance:
            text_embeddings = torch.cat([text_embeddings, guidance_3d, guidance_traj], dim=1)
            print("input guidance concat embeddings :", text_embeddings.shape)

        # define initial latents
        latents = self.image2latent(image)

        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.:
            max_length = text_input.input_ids.shape[-1]
            unconditional_input = self.tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]

            if use_guidance:
                guidance_3d_length = guidance_3d.shape[-2]
                unconditional_input_3d = np.zeros((guidance_3d_length * 8, 3))
                unconditional_input_3d_tensor = torch.tensor(unconditional_input_3d, device=DEVICE, dtype=torch.float32)
                unconditional_embeddings_3d = self.spatial_guidance_model(unconditional_input_3d_tensor)
                unconditional_embeddings_3d = unconditional_embeddings_3d.repeat(batch_size, 1, 1)

                guidance_traj_length = guidance_traj.shape[-2]
                unconditional_input_traj = np.zeros((guidance_traj_length * 2, 7))
                unconditional_input_traj_tensor = torch.tensor(unconditional_input_traj, device=DEVICE, dtype=torch.float32)
                unconditional_embeddings_traj = self.trajectory_guidance_model(unconditional_input_traj_tensor)
                unconditional_embeddings_traj = unconditional_embeddings_traj.repeat(batch_size, 1, 1)

                unconditional_embeddings = torch.cat([unconditional_embeddings, unconditional_embeddings_3d, unconditional_embeddings_traj], dim=1)

            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)
            print("cfg text embeddings shape: ", text_embeddings.shape)

        print("latents shape: ", latents.shape)
        # interative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        print("Valid timesteps: ", reversed(self.scheduler.timesteps))
        # print("attributes: ", self.scheduler.__dict__)
        latents_list = [latents]
        pred_x0_list = [latents]
        for i, t in enumerate(tqdm(reversed(self.scheduler.timesteps), desc="DDIM Inversion")):
            if num_actual_inference_steps is not None and i >= num_actual_inference_steps:
                continue

            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents

            # predict the noise
            noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings)
            if guidance_scale > 1.:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            # compute the previous noise sample x_t-1 -> x_t
            latents, pred_x0 = self.inv_step(noise_pred, t, latents)
            latents_list.append(latents)
            pred_x0_list.append(pred_x0)

        if return_intermediates:
            # return the intermediate laters during inversion
            # pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
            return latents, pred_x0_list
        return latents

    @torch.no_grad()
    def invert_nvs(
        self,
        image: torch.Tensor,
        prompt,
        guidance_3d,
        guidance_traj,
        num_inference_steps=50,
        num_actual_inference_steps=None,
        guidance_scale=[7.5,7.5,7.5],
        eta=0.0,
        return_intermediates=False,
        **kwds):
        """
        invert a real image into noise map with determinisc DDIM inversion
        """

        if isinstance(guidance_scale, float):
            guidance_scale = [guidance_scale, guidance_scale, guidance_scale]

        if guidance_scale[0] > 1. or guidance_scale[1] > 1. or guidance_scale[2] > 1.:
            perfrom_cfg = True
        else:
            perfrom_cfg = False

        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        batch_size = image.shape[0]
        if isinstance(prompt, list):
            if batch_size == 1:
                image = image.expand(len(prompt), -1, -1, -1)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        if batch_size > 1:
            guidance_3d = guidance_3d.repeat(batch_size, 1, 1)
            guidance_traj = guidance_traj.repeat(batch_size, 1, 1)

        # text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        print("input text embeddings :", text_embeddings.shape)
        # define initial latents
        latents = self.image2latent(image)

        # unconditional embedding for classifier free guidance
        if perfrom_cfg:
        # if guidance_scale > 1.:
            max_length = text_input.input_ids.shape[-1]
            unconditional_input_text = self.tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            unconditional_embeddings_text = self.text_encoder(unconditional_input_text.input_ids.to(DEVICE))[0]
            text_embeddings = torch.cat([unconditional_embeddings_text, text_embeddings, text_embeddings, text_embeddings], dim=0)

            guidance_3d_length = guidance_3d.shape[-2]
            unconditional_input_3d = np.zeros((guidance_3d_length * 8, 3))
            unconditional_input_3d_tensor = torch.tensor(unconditional_input_3d, device=DEVICE, dtype=torch.float32)
            unconditional_embeddings_3d = self.spatial_guidance_model(unconditional_input_3d_tensor)
            unconditional_embeddings_3d = unconditional_embeddings_3d.repeat(batch_size, 1, 1)
            spatial_embeddings = torch.cat([unconditional_embeddings_3d, unconditional_embeddings_3d, guidance_3d, guidance_3d], dim=0)

            guidance_traj_length = guidance_traj.shape[-2]
            unconditional_input_traj = np.zeros((guidance_traj_length * 2, 7))
            unconditional_input_traj_tensor = torch.tensor(unconditional_input_traj, device=DEVICE, dtype=torch.float32)
            unconditional_embeddings_traj = self.trajectory_guidance_model(unconditional_input_traj_tensor)
            unconditional_embeddings_traj = unconditional_embeddings_traj.repeat(batch_size, 1, 1)
            trajectory_embeddings = torch.cat([unconditional_embeddings_traj, unconditional_embeddings_traj, unconditional_embeddings_traj, guidance_traj], dim=0)

            print("text_embeddings shape: ", text_embeddings.shape)
            print("spatial_embeddings shape: ", spatial_embeddings.shape)            
            print("trajectory_embeddings shape: ", trajectory_embeddings.shape)

        added_cond_kwargs = {
            "guidance_3d": guidance_3d,
            "guidance_traj": guidance_traj
        }

        print("latents shape: ", latents.shape)
        # interative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        print("Valid timesteps: ", reversed(self.scheduler.timesteps))
        # print("attributes: ", self.scheduler.__dict__)
        latents_list = [latents]
        pred_x0_list = [latents]
        for i, t in enumerate(tqdm(reversed(self.scheduler.timesteps), desc="DDIM Inversion")):
            if num_actual_inference_steps is not None and i >= num_actual_inference_steps:
                continue

            if perfrom_cfg:
            # if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 4)
            else:
                model_inputs = latents

            # predict the noise
            noise_pred = self.unet(
                model_inputs, 
                t, 
                encoder_hidden_states=text_embeddings,
                added_cond_kwargs=added_cond_kwargs
                )
            if perfrom_cfg:
            # if guidance_scale > 1.:
                noise_pred_uncon, noise_pred_con1, noise_pred_con2, noise_pred_con3 = noise_pred.chunk(4, dim=0)
                noise_pred = noise_pred_uncon \
                                + guidance_scale[0] * (noise_pred_con1 - noise_pred_uncon) \
                                + guidance_scale[1] * (noise_pred_con2 - noise_pred_con1) \
                                + guidance_scale[2] * (noise_pred_con3 - noise_pred_con2)                                
                
            # compute the previous noise sample x_t-1 -> x_t
            latents, pred_x0 = self.inv_step(noise_pred, t, latents)
            latents_list.append(latents)
            pred_x0_list.append(pred_x0)

        if return_intermediates:
            # return the intermediate laters during inversion
            # pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
            return latents, pred_x0_list
        return latents