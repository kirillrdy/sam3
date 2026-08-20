const std = @import("std");

pub const SAM3Config = struct {
    image_size: usize = 256,
    patch_size: usize = 16,
    encoder_embed_dim: usize = 256,
    encoder_num_heads: usize = 8,
    encoder_depth: usize = 4,
    encoder_mlp_ratio: f32 = 4.0,

    prompt_embed_dim: usize = 256,
    text_vocab_size: usize = 1024,
    text_max_seq_len: usize = 32,
    text_num_layers: usize = 2,

    presence_threshold: f32 = 0.5,
    presence_head_dim: usize = 256,
    num_object_queries: usize = 16,

    mask_decoder_num_heads: usize = 8,
    mask_decoder_depth: usize = 2,
    mask_decoder_num_multimask_outputs: usize = 3,
    mask_decoder_iou_head_depth: usize = 3,

    memory_bank_size: usize = 7,
    memory_dim: usize = 64,
    num_maskmem_features: usize = 64,

    pub fn sam3_tiny() SAM3Config {
        return SAM3Config{
            .image_size = 128,
            .patch_size = 16,
            .encoder_embed_dim = 64,
            .encoder_num_heads = 4,
            .encoder_depth = 2,
            .encoder_mlp_ratio = 2.0,
            .prompt_embed_dim = 64,
            .text_vocab_size = 512,
            .text_max_seq_len = 16,
            .text_num_layers = 1,
            .presence_threshold = 0.5,
            .presence_head_dim = 64,
            .num_object_queries = 8,
            .mask_decoder_num_heads = 4,
            .mask_decoder_depth = 1,
            .mask_decoder_num_multimask_outputs = 3,
            .mask_decoder_iou_head_depth = 2,
            .memory_bank_size = 5,
            .memory_dim = 32,
            .num_maskmem_features = 32,
        };
    }

    pub fn sam3_base() SAM3Config {
        return SAM3Config{
            .image_size = 512,
            .patch_size = 16,
            .encoder_embed_dim = 256,
            .encoder_num_heads = 8,
            .encoder_depth = 6,
            .encoder_mlp_ratio = 4.0,
            .prompt_embed_dim = 256,
            .text_vocab_size = 2048,
            .text_max_seq_len = 32,
            .text_num_layers = 3,
            .presence_threshold = 0.5,
            .presence_head_dim = 256,
            .num_object_queries = 32,
            .mask_decoder_num_heads = 8,
            .mask_decoder_depth = 2,
            .mask_decoder_num_multimask_outputs = 3,
            .mask_decoder_iou_head_depth = 3,
            .memory_bank_size = 7,
            .memory_dim = 64,
            .num_maskmem_features = 64,
        };
    }

    /// Dimensions of Meta's released SAM 3 checkpoint (`facebook/sam3`,
    /// 859.9M parameters). Verified against the checkpoint header by
    /// `sam3 weights <file> --verify`.
    pub fn sam3_full() SAM3Config {
        return SAM3Config{
            .image_size = 1008,
            .patch_size = 14,
            .encoder_embed_dim = 1024,
            .encoder_num_heads = 16,
            .encoder_depth = 32,
            // The vision backbone MLP widens 1024 -> 4736.
            .encoder_mlp_ratio = 4.625,
            .prompt_embed_dim = 256,
            .text_vocab_size = 49408,
            .text_max_seq_len = 32,
            .text_num_layers = 24,
            .presence_threshold = 0.5,
            .presence_head_dim = 256,
            .num_object_queries = 200,
            .mask_decoder_num_heads = 8,
            .mask_decoder_depth = 6,
            .mask_decoder_num_multimask_outputs = 3,
            .mask_decoder_iou_head_depth = 3,
            .memory_bank_size = 7,
            .memory_dim = 64,
            .num_maskmem_features = 64,
        };
    }

    pub inline fn gridH(self: SAM3Config) usize {
        return self.image_size / self.patch_size;
    }

    pub inline fn gridW(self: SAM3Config) usize {
        return self.image_size / self.patch_size;
    }

    pub inline fn numPatches(self: SAM3Config) usize {
        return self.gridH() * self.gridW();
    }
};
