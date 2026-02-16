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
#define FRAMES_PER_SEGMENT  10    /* render 10 frames per segment for crossfade */
#define TOTAL_SEGMENTS      658   /* ceil(6572 / 10) */

/* Time values: sweep [0..1023] mapping to t ∈ [-1.0, +1.0] via RTL.
 * With 10 frames: time_step[fi] = fi * 1023 / 9 (evenly spaced, 0 to 1023).
 * RTL maps: time_val = time_step * (2/1024) - 1.0 in Q4.28. */
static const u16 time_steps[FRAMES_PER_SEGMENT] = {
    0, 114, 227, 341, 455, 568, 682, 796, 909, 1023
};

/* Alpha ramp: 0, 28, 57, 85, 113, 142, 170, 198, 227, 255 */
static const u8 alpha_ramp[FRAMES_PER_SEGMENT] = {
    0, 28, 57, 85, 113, 142, 170, 198, 227, 255
};

/* ================================================================
 * SD card weight buffer (in DDR)
 * ================================================================ */
static u32 weight_buf[N_PARAMS];

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

static int load_segment_from_sd(FIL *fp, const char *filename, u32 *buf)
{
    FRESULT res;
    UINT br;

    res = f_open(fp, filename, FA_READ);
    if (res != FR_OK) {
        xil_printf("ERR: open %s failed (%d)\r\n", filename, res);
        return -1;
    }

    res = f_read(fp, buf, N_PARAMS * 4, &br);
    f_close(fp);

    if (res != FR_OK || br != N_PARAMS * 4) {
        xil_printf("ERR: read %s: res=%d br=%u (expected %u)\r\n",
                    filename, res, br, N_PARAMS * 4);
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
    char filename[32];
    int seg;
    int frame;
    int cur_bank;  /* 0 = next load goes to A, 1 = next load goes to B */

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

    /* Enable PS override + threshold mode */
    Xil_Out32(REG_CTRL, CTRL_PS_OVERRIDE | CTRL_THRESHOLD_EN);

    /* Wait for any in-progress PL frame to complete, then drain stale frame_done */
    while (Xil_In32(REG_STATUS) & STATUS_FRAME_BUSY)
        ;
    (void)Xil_In32(REG_STATUS);  /* Clear stale frame_done_sticky */

    xil_printf("PS override enabled, threshold ON.\r\n");

    /* Load first two segments into BRAM A and B */
    xil_printf("Loading seg_000...\r\n");
    if (load_segment_from_sd(&fil, "seg_000.bin", weight_buf) < 0) {
        xil_printf("Halting.\r\n");
        while (1) ;
    }
    write_weights(WEIGHT_A_BASE, weight_buf, N_PARAMS);

    xil_printf("Loading seg_001...\r\n");
    if (load_segment_from_sd(&fil, "seg_001.bin", weight_buf) < 0) {
        xil_printf("Halting.\r\n");
        while (1) ;
    }
    write_weights(WEIGHT_B_BASE, weight_buf, N_PARAMS);

    xil_printf("Starting playback...\r\n");

    /*
     * Playback loop:
     *   - Crossfade from current bank A→B over FRAMES_PER_SEGMENT frames
     *   - Load next segment into the bank that was just faded out (A)
     *   - Swap: now crossfade B→A
     *   - Repeat
     *
     * Bank tracking:
     *   cur_bank=0: A has old segment, B has new → alpha ramps 0→255
     *   cur_bank=1: B has old segment, A has new → alpha ramps 255→0
     */
    cur_bank = 0;

    for (seg = 0; seg < TOTAL_SEGMENTS - 1; seg++) {
        /* Crossfade: render FRAMES_PER_SEGMENT frames */
        for (frame = 0; frame < FRAMES_PER_SEGMENT; frame++) {
            u8 alpha;
            if (cur_bank == 0) {
                /* Fading from A (old) to B (new): alpha goes 0→255 */
                alpha = alpha_ramp[frame];
            } else {
                /* Fading from B (old) to A (new): alpha goes 255→0 */
                alpha = 255 - alpha_ramp[frame];
            }
            render_frame(time_steps[frame], alpha);
        }

        /* Load next segment into the bank that just became "old" */
        int next_seg = seg + 2;
        if (next_seg < TOTAL_SEGMENTS) {
            /* Format seg_NNN.bin without snprintf (standalone BSP may lack it) */
            filename[0] = 's'; filename[1] = 'e'; filename[2] = 'g'; filename[3] = '_';
            filename[4] = '0' + (next_seg / 100) % 10;
            filename[5] = '0' + (next_seg / 10) % 10;
            filename[6] = '0' + next_seg % 10;
            filename[7] = '.'; filename[8] = 'b'; filename[9] = 'i'; filename[10] = 'n';
            filename[11] = '\0';

            if (load_segment_from_sd(&fil, filename, weight_buf) < 0) {
                xil_printf("Segment %d load failed, looping last.\r\n", next_seg);
                break;
            }

            if (cur_bank == 0) {
                /* A was old, B was new. Load next into A. */
                write_weights(WEIGHT_A_BASE, weight_buf, N_PARAMS);
            } else {
                /* B was old, A was new. Load next into B. */
                write_weights(WEIGHT_B_BASE, weight_buf, N_PARAMS);
            }
        }

        /* Swap active bank */
        cur_bank = 1 - cur_bank;

        if ((seg & 0x1F) == 0) {
            xil_printf("Seg %d/%d\r\n", seg, TOTAL_SEGMENTS);
        }
    }

    /* Render final segment fully (all frames at alpha=255 or 0) */
    xil_printf("Final segment, holding...\r\n");
    for (frame = 0; frame < FRAMES_PER_SEGMENT; frame++) {
        u8 alpha = (cur_bank == 0) ? 255 : 0;
        render_frame(time_steps[frame], alpha);
    }

    /* Loop forever on last frame */
    xil_printf("Playback complete. Looping.\r\n");
    while (1) {
        usleep(1000000);
    }

    return 0;
}
