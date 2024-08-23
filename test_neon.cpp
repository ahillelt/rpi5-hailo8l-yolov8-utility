#include <arm_neon.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>

// NEON-optimized normalization function
void normalizeImageNeon(uint8_t* input, float* output, int size) {
    const float32x4_t scale = vdupq_n_f32(1.0f / 255.0f);

    // Ensure output is aligned
    if (reinterpret_cast<uintptr_t>(output) % 16 != 0) {
        std::cerr << "Output array is not aligned!" << std::endl;
        return;
    }

    int i = 0;
    for (; i <= size - 16; i += 16) {
        uint8x16_t in = vld1q_u8(input + i);

        // Convert and scale low and high parts of 16 bytes
        float32x4_t f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vget_low_u8(in)))));
        float32x4_t f1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vmovl_u8(vget_low_u8(in)))));
        float32x4_t f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vget_high_u8(in)))));
        float32x4_t f3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vmovl_u8(vget_high_u8(in)))));

        f0 = vmulq_f32(f0, scale);
        f1 = vmulq_f32(f1, scale);
        f2 = vmulq_f32(f2, scale);
        f3 = vmulq_f32(f3, scale);

        // Store results
        vst1q_f32(output + i, f0);
        vst1q_f32(output + i + 4, f1);
        vst1q_f32(output + i + 8, f2);
        vst1q_f32(output + i + 12, f3);
    }

    // Handle any remaining elements
    for (; i < size; ++i) {
        output[i] = input[i] * (1.0f / 255.0f);
    }
}

int main() {
    int size = 1228800;
    uint8_t* input = new uint8_t[size];
    float* output;

    // Allocate aligned memory for output
    if (posix_memalign(reinterpret_cast<void**>(&output), 16, size * sizeof(float)) != 0) {
        std::cerr << "Failed to allocate aligned memory for output!" << std::endl;
        return 1;
    }

    // Initialize input with some values for testing
    for (int i = 0; i < size; ++i) {
        input[i] = static_cast<uint8_t>(i % 256);
    }

    normalizeImageNeon(input, output, size);

    // Clean up
    delete[] input;
    free(output);

    return 0;
}