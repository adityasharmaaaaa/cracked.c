u32 idx_be32(const u8 bytes[4]){
    return ((u32)bytes[0] << 24) | ((u32)bytes[1] << 16) | ((u32)bytes[2] << 8) | (u32)bytes[3];
}

b32 idx_read_header(FILE* f, idx_header* out) {
    u8 magic[4];
    if (fread(magic, 1, 4, f) != 4) { return false; }
 
    if (magic[0] != 0 || magic[1] != 0) { return false; }      
    if (magic[2] != 0x08) { return false; }                    
 
    u32 num_dims = magic[3];
    if (num_dims == 0 || num_dims > IDX_MAX_DIMS) { return false; }
 
    idx_header header = { 0 };
    header.num_dims = num_dims;
    for (u32 i = 0; i < num_dims; i++) {
        u8 word[4];
        if (fread(word, 1, 4, f) != 4) { return false; }       
        header.dims[i] = idx_be32(word);
    }
    header.data_offset = 4 + 4 * (u64)num_dims;
 
    *out = header;
    return true;
}