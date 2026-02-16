/*
 * mlp_loader.c — Bare-metal Bad Apple playback on Zynq MLP inference engine
 *
 * Reads trained SIREN weight segments from SD card (FAT32) and pushes them
 * to the PL inference engine via AXI-Lite. The PL renders each frame in
 * real-time using 18 parallel neural inference cores.
 *
 * Boot chain: FSBL → bitstream → this application (loaded to DDR by FSBL)
 *
 * Build (after generating BSP from XSA):
 *   arm-none-eabi-gcc -mcpu=cortex-a9 -mfpu=vfpv3 -mfloat-abi=hard \
 *     -O2 -I<bsp>/include -L<bsp>/lib \
 *     -Wl,-T -Wl,lscript.ld \
 *     -o mlp_loader.elf mlp_loader.c \
 *     -lxilffs -lxil -lgcc -lc
 */

#include "xil_io.h"
#include "xil_printf.h"
#include "ff.h"
#include "sleep.h"
#include <string.h>

/* ================================================================
 * AXI-Lite register map (base 0x43C00000)
 * ================================================================ */
#define AXI_BASE        0x43C00000U

/* Weight BRAM regions */
#define WEIGHT_A_BASE   (AXI_BASE + 0x0000)  /* word[i] at offset i*4 */
#define WEIGHT_B_BASE   (AXI_BASE + 0x2000)

/* Control registers */
#define REG_CTRL        (AXI_BASE + 0x4000)
#define REG_MORPH_ALPHA (AXI_BASE + 0x4004)
#define REG_TIME_VAL    (AXI_BASE + 0x4008)
#define REG_STATUS      (AXI_BASE + 0x400C)
#define REG_FRAME_START (AXI_BASE + 0x4010)

/* CTRL register bits */
#define CTRL_PS_OVERRIDE   (1U << 0)
#define CTRL_THRESHOLD_EN  (1U << 1)
#define CTRL_FRAME_AUTO    (1U << 2)

/* STATUS register bits */
#define STATUS_FRAME_BUSY  (1U << 0)
#define STATUS_FRAME_DONE  (1U << 1)

/* ================================================================
 * Network geometry (must match RTL)
 * ================================================================ */
#define N_HIDDEN    32
#define N_INPUT     3
#define N_OUTPUT    3

/* Total parameters: (3*32+32) + (32*32+32) + (32*3+3) = 128 + 1056 + 99 = 1283 */
#define N_PARAMS    1283

/* ================================================================
 * Playback parameters
 * ================================================================ */
#define FRAMES_PER_SEGMENT  3     /* 3 frames/segment for ~real-time at 8 FPS */
#define TOTAL_SEGMENTS      658   /* ceil(6572 / 10) — trained with 10 frames */

/* Time values: sweep [0..1023] mapping to t ∈ [-1.0, +1.0] via RTL.
 * With 3 frames: sample start, middle, and end of each segment's time range.
 * RTL maps: time_val = time_step * (2/1024) - 1.0 in Q4.28.
 *
 * NOTE: Weight morphing (alpha blending between banks) was tried but produces
 * distorted midpoint frames — SIREN weight-space interpolation is non-smooth
 * for independently-trained networks due to sin() activations. Hard cuts
 * between segments work better for video playback. */
static const u16 time_steps[FRAMES_PER_SEGMENT] = {
    0, 512, 1023
};

/* ================================================================
 * SD card weight buffer (in DDR)
 * ================================================================ */
static u32 weight_buf[N_PARAMS];

/* Segment size in bytes */
#define SEGMENT_BYTES   (N_PARAMS * 4)

/* ================================================================
 * Helper functions
 * ================================================================ */

static void write_weights(u32 bank_base, const u32 *weights, int count)
{
    int i;
    for (i = 0; i < count; i++) {
        Xil_Out32(bank_base + (i << 2), weights[i]);
    }
}

static int load_segment(FIL *fp, int seg_num, u32 *buf)
{
    FRESULT res;
    UINT br;

    res = f_lseek(fp, (FSIZE_t)seg_num * SEGMENT_BYTES);
    if (res != FR_OK) {
        xil_printf("ERR: seek seg %d failed (%d)\r\n", seg_num, res);
        return -1;
    }

    res = f_read(fp, buf, SEGMENT_BYTES, &br);
    if (res != FR_OK || br != SEGMENT_BYTES) {
        xil_printf("ERR: read seg %d: res=%d br=%u (expected %u)\r\n",
                    seg_num, res, br, SEGMENT_BYTES);
        return -1;
    }

    return 0;
}

static void wait_frame_done(void)
{
    /* Poll STATUS until frame_done is set */
    while (!(Xil_In32(REG_STATUS) & STATUS_FRAME_DONE))
        ;
}

static void trigger_frame(void)
{
    /* Write any value to FRAME_START to trigger a new frame */
    Xil_Out32(REG_FRAME_START, 1);
}

static void render_frame(u16 time_val, u8 alpha)
{
    Xil_Out32(REG_TIME_VAL, time_val);
    Xil_Out32(REG_MORPH_ALPHA, alpha);
    trigger_frame();
    wait_frame_done();
}

/* ================================================================
 * Main
 * ================================================================ */
int main(void)
{
    FATFS fs;
    FIL fil;
    FRESULT res;
    int seg;
    int frame;

    xil_printf("\r\n=== Bad Apple Neural Video Codec ===\r\n");
    xil_printf("SIREN 3->%d->%d->3, %d params, %d segments\r\n",
               N_HIDDEN, N_HIDDEN, N_PARAMS, TOTAL_SEGMENTS);

    /* Mount SD card */
    res = f_mount(&fs, "0:", 1);
    if (res != FR_OK) {
        xil_printf("ERR: SD mount failed (%d)\r\n", res);
        xil_printf("Halting.\r\n");
        while (1) ;
    }
    xil_printf("SD card mounted.\r\n");

    /* Open single weights file (all segments concatenated) */
    res = f_open(&fil, "weights.bin", FA_READ);
    if (res != FR_OK) {
        xil_printf("ERR: open weights.bin failed (%d)\r\n", res);
        xil_printf("Halting.\r\n");
        while (1) ;
    }
    xil_printf("weights.bin opened (%lu bytes).\r\n", f_size(&fil));

    /* Enable PS override (no threshold — show raw grayscale) */
    Xil_Out32(REG_CTRL, CTRL_PS_OVERRIDE);

    /* Wait for any in-progress PL frame to complete, then drain stale frame_done */
    while (Xil_In32(REG_STATUS) & STATUS_FRAME_BUSY)
        ;
    (void)Xil_In32(REG_STATUS);  /* Clear stale frame_done_sticky */

    xil_printf("PS override enabled.\r\n");

    /* Load first segment into BRAM A */
    xil_printf("Loading segment 0...\r\n");
    if (load_segment(&fil, 0, weight_buf) < 0) {
        xil_printf("Halting.\r\n");
        while (1) ;
    }
    write_weights(WEIGHT_A_BASE, weight_buf, N_PARAMS);

    xil_printf("Starting playback...\r\n");

    for (seg = 0; seg < TOTAL_SEGMENTS; seg++) {
        /* Render all frames for this segment using bank A (alpha=0) */
        for (frame = 0; frame < FRAMES_PER_SEGMENT; frame++) {
            render_frame(time_steps[frame], 0);
        }

        /* Load next segment into bank A */
        if (seg + 1 < TOTAL_SEGMENTS) {
            if (load_segment(&fil, seg + 1, weight_buf) < 0) {
                xil_printf("Segment %d load failed, stopping.\r\n", seg + 1);
                break;
            }
            write_weights(WEIGHT_A_BASE, weight_buf, N_PARAMS);
        }

        if ((seg & 0x1F) == 0) {
            xil_printf("Seg %d/%d\r\n", seg, TOTAL_SEGMENTS);
        }
    }

    f_close(&fil);

    /* Loop forever on last frame */
    xil_printf("Playback complete. Looping.\r\n");
    while (1) {
        usleep(1000000);
    }

    return 0;
}
