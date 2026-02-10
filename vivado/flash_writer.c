/* flash_writer.c — Standalone QSPI flash programmer for Zynq-7000
 *
 * Uses the Zynq PS QSPI controller in generic (IO) mode to program
 * the onboard QSPI flash. Runs from OCM after FSBL initializes the PS.
 *
 * Zynq QSPI transfers 4 bytes at a time via 32-bit TX/RX FIFOs.
 * TX FIFO depth: 128 bytes (32 entries), RX FIFO depth: 128 bytes.
 *
 * Protocol via control block at 0x00020000 (OCM):
 *   [0] command:    1=erase+write, 2=verify, 0=idle
 *   [1] src_addr:   DDR address of data
 *   [2] length:     bytes to write
 *   [3] status:     0=busy, 1=done, 0xFF=error
 *   [4] progress:   bytes processed
 *   [5] error_code: detail
 *   [6] jedec_id:   flash JEDEC ID
 *
 * Compile: arm-none-eabi-gcc -mcpu=cortex-a9 -mfloat-abi=soft -O2
 *          -fno-builtin -nostdlib -Ttext=0x0
 *          -o flash_writer.elf flash_writer.c
 */

#include <stdint.h>
#include <stddef.h>

/* GCC may emit implicit calls to these even with -fno-builtin */
void *memset(void *s, int c, size_t n) {
    unsigned char *p = s;
    while (n--) *p++ = (unsigned char)c;
    return s;
}

void *memcpy(void *dest, const void *src, size_t n) {
    unsigned char *d = dest;
    const unsigned char *s = src;
    while (n--) *d++ = *s++;
    return dest;
}

/* ---- Zynq QSPI Controller Registers (base 0xE000D000) ---- */
#define QSPI_BASE   0xE000D000u

#define XQSPIPS_CR      (*(volatile uint32_t *)(QSPI_BASE + 0x00))  /* Config */
#define XQSPIPS_SR      (*(volatile uint32_t *)(QSPI_BASE + 0x04))  /* Status */
#define XQSPIPS_IER     (*(volatile uint32_t *)(QSPI_BASE + 0x08))  /* IRQ Enable */
#define XQSPIPS_IDR     (*(volatile uint32_t *)(QSPI_BASE + 0x0C))  /* IRQ Disable */
#define XQSPIPS_ER      (*(volatile uint32_t *)(QSPI_BASE + 0x14))  /* Enable */
#define XQSPIPS_DR      (*(volatile uint32_t *)(QSPI_BASE + 0x18))  /* Delay */
#define XQSPIPS_TXD0    (*(volatile uint32_t *)(QSPI_BASE + 0x1C))  /* TX 4-byte */
#define XQSPIPS_RXD     (*(volatile uint32_t *)(QSPI_BASE + 0x20))  /* RX */
#define XQSPIPS_SICR    (*(volatile uint32_t *)(QSPI_BASE + 0x24))  /* Slave idle */
#define XQSPIPS_TXTH    (*(volatile uint32_t *)(QSPI_BASE + 0x28))  /* TX threshold */
#define XQSPIPS_RXTH    (*(volatile uint32_t *)(QSPI_BASE + 0x2C))  /* RX threshold */
#define XQSPIPS_TXD1    (*(volatile uint32_t *)(QSPI_BASE + 0x80))  /* TX 1-byte */
#define XQSPIPS_TXD2    (*(volatile uint32_t *)(QSPI_BASE + 0x84))  /* TX 2-byte */
#define XQSPIPS_TXD3    (*(volatile uint32_t *)(QSPI_BASE + 0x88))  /* TX 3-byte */
#define XQSPIPS_LQSPI   (*(volatile uint32_t *)(QSPI_BASE + 0xA0))  /* LQSPI Config */

/* CR bit definitions (from Zynq TRM UG585 Table 12-6 / xqspips_hw.h) */
#define CR_IFMODE       (1u << 31)  /* Flash interface mode (LQSPI) */
#define CR_HOLDB_DR     (1u << 19)  /* Drive HOLD#/WP# high */
#define CR_MAN_START    (1u << 16)  /* Manual start transfer (write-1-to-start) */
#define CR_MAN_START_EN (1u << 15)  /* Enable manual start mode */
#define CR_SS_FORCE     (1u << 14)  /* Manual chip select */
#define CR_PCS          (1u << 9)   /* Peripheral chip select decode */
#define CR_REF_CLK      (1u << 8)   /* Reference clock select */
#define CR_BAUD_DIV(n)  (((n) & 7) << 3) /* Baud rate: /2^(n+1) */
#define CR_CPHA         (1u << 2)   /* Clock phase */
#define CR_CPOL         (1u << 1)   /* Clock polarity */
#define CR_MSTREN       (1u << 0)   /* Master mode enable */

/* SR bit definitions */
#define SR_TX_FULL      (1u << 3)   /* TX FIFO full */
#define SR_TX_NOT_FULL  (1u << 2)   /* TX FIFO not full */
#define SR_RX_NOT_EMPTY (1u << 4)   /* RX FIFO not empty */
#define SR_TX_UNDERFLOW (1u << 6)   /* TX underflow */

/* Flash commands (3-byte address for W25Q128/16MB) */
#define CMD_WRITE_ENABLE     0x06
#define CMD_READ_STATUS      0x05
#define CMD_READ_STATUS2     0x35
#define CMD_WRITE_STATUS2    0x31
#define CMD_READ_JEDEC       0x9F
#define CMD_PAGE_PROGRAM     0x02   /* 3-byte addr page program */
#define CMD_READ_DATA        0x03   /* 3-byte addr read */
#define CMD_BLOCK_ERASE_64K  0xD8   /* 3-byte addr 64KB block erase */
#define CMD_SECTOR_ERASE_4K  0x20   /* 3-byte addr 4KB sector erase */

#define PAGE_SIZE   256
#define BLOCK_SIZE  65536

/* Control block in OCM */
volatile uint32_t *ctrl = (volatile uint32_t *)0x00020000;

static void delay_us(int us) {
    /* Rough delay at ~650 MHz */
    volatile int i;
    for (i = 0; i < us * 100; i++) __asm__ volatile("");
}

/* Drain the RX FIFO */
static void qspi_drain_rx(void) {
    while (XQSPIPS_SR & SR_RX_NOT_EMPTY)
        (void)XQSPIPS_RXD;
}

/* Do a single QSPI transfer: write n_tx bytes, read n_tx bytes back.
 * Uses the appropriate TXDn register for transfer size.
 * tx_data is packed little-endian: byte0 in bits[7:0], etc.
 * Returns received data (little-endian packed). */
static uint32_t qspi_xfer(uint32_t tx_data, int nbytes) {
    qspi_drain_rx();

    /* Write to appropriate TX register for transfer size */
    switch (nbytes) {
        case 1: XQSPIPS_TXD1 = tx_data; break;
        case 2: XQSPIPS_TXD2 = tx_data; break;
        case 3: XQSPIPS_TXD3 = tx_data; break;
        default: XQSPIPS_TXD0 = tx_data; break; /* 4 bytes */
    }

    /* Trigger transfer */
    XQSPIPS_CR |= CR_MAN_START;

    /* Wait for TX complete (RX data available) */
    while (!(XQSPIPS_SR & SR_RX_NOT_EMPTY)) ;

    return XQSPIPS_RXD;
}

/* Multi-word QSPI transfer for longer commands.
 * Uses SS_FORCE to hold CS asserted across multiple FIFO fills.
 * TX FIFO is 32 entries (128 bytes), so transfers > 128 bytes
 * require multiple fill/drain cycles with CS held low. */
static void qspi_xfer_multi(const uint32_t *tx, uint32_t *rx, int nwords) {
    int tx_idx = 0, rx_idx = 0;
    qspi_drain_rx();

    /* Force CS low for the entire transfer */
    XQSPIPS_CR |= CR_SS_FORCE;

    while (tx_idx < nwords || rx_idx < nwords) {
        /* Fill TX FIFO (up to 32 entries per batch) */
        int batch = 0;
        while (tx_idx < nwords && batch < 32) {
            XQSPIPS_TXD0 = tx[tx_idx++];
            batch++;
        }

        if (batch > 0) {
            /* Start this batch */
            XQSPIPS_CR |= CR_MAN_START;

            /* Read RX for this batch */
            int i;
            for (i = 0; i < batch; i++) {
                while (!(XQSPIPS_SR & SR_RX_NOT_EMPTY)) ;
                rx[rx_idx++] = XQSPIPS_RXD;
            }
        }
    }

    /* Release CS */
    XQSPIPS_CR &= ~CR_SS_FORCE;
}

static void qspi_init(void) {
    /* Disable controller */
    XQSPIPS_ER = 0;
    delay_us(100);

    /* Disable LQSPI mode */
    XQSPIPS_LQSPI &= ~(1u << 31);

    /* Configure: master mode, manual start, HOLD# driven, baud /4, CPOL=0, CPHA=0 */
    XQSPIPS_CR = CR_HOLDB_DR | CR_MAN_START_EN | CR_BAUD_DIV(1) | CR_MSTREN;

    /* Disable all interrupts */
    XQSPIPS_IDR = 0x7F;

    /* Set thresholds */
    XQSPIPS_TXTH = 1;
    XQSPIPS_RXTH = 1;

    /* Enable controller */
    XQSPIPS_ER = 1;
    delay_us(100);

    /* Drain any stale RX data */
    qspi_drain_rx();
}

/* Read JEDEC ID */
static uint32_t flash_read_jedec(void) {
    /* Send: 9F FF FF FF (cmd + 3 dummy) → receive: XX MFR TYP CAP */
    uint32_t rx = qspi_xfer(0xFFFFFF9F, 4);  /* little-endian: 9F is byte0 */
    /* rx is little-endian: byte0=junk, byte1=mfr, byte2=type, byte3=cap */
    uint32_t mfr = (rx >> 8) & 0xFF;
    uint32_t typ = (rx >> 16) & 0xFF;
    uint32_t cap = (rx >> 24) & 0xFF;
    return (mfr << 16) | (typ << 8) | cap;
}

/* Send 1-byte command (write enable, etc.)
 * Must use TXD1 for exact 1-byte CS assertion. */
static void flash_cmd1(uint8_t cmd) {
    qspi_drain_rx();
    XQSPIPS_TXD1 = (uint32_t)cmd;
    XQSPIPS_CR |= CR_MAN_START;
    while (!(XQSPIPS_SR & SR_RX_NOT_EMPTY)) ;
    (void)XQSPIPS_RXD;
}

/* Read status register — uses 4-byte TXD0 transfer for reliability */
static uint8_t flash_read_status(void) {
    uint32_t rx = qspi_xfer(0xFFFFFF05, 4); /* CMD_READ_STATUS=0x05, then 3xFF */
    return (rx >> 8) & 0xFF; /* status is in byte 1 */
}

/* Read status register 2 (contains QE bit) */
static uint8_t flash_read_status2(void) {
    uint32_t rx = qspi_xfer(0xFFFFFF35, 4); /* CMD_READ_STATUS2=0x35 */
    return (rx >> 8) & 0xFF;
}

/* Wait for flash WIP (Write In Progress) to clear */
static int flash_wait_ready(int timeout_ms) {
    int i;
    for (i = 0; i < timeout_ms; i++) {
        if (!(flash_read_status() & 0x01)) return 0;
        delay_us(1000);
    }
    return -1; /* timeout */
}

/* Enable Quad mode (set QE bit in status register 2).
 * QE is non-volatile — persists across power cycles.
 * Required for BootROM which uses Quad Output Fast Read (0x6B). */
static void flash_enable_quad(void) {
    uint8_t sr2 = flash_read_status2();
    if (sr2 & 0x02) return; /* QE already set */

    flash_cmd1(CMD_WRITE_ENABLE);
    delay_us(10);

    /* Write Status Register 2: CMD 0x31 + data byte with QE=1
     * Must be exactly 2 bytes — extra clocks prevent latch */
    qspi_xfer(0x0231, 2); /* TXD2: 0x31=cmd, 0x02=QE bit */
    flash_wait_ready(100);
}

/* Erase 64KB block (3-byte address) */
static int flash_erase_block(uint32_t addr) {
    flash_cmd1(CMD_WRITE_ENABLE);
    delay_us(10);

    /* D8 AA AA AA (cmd + 3-byte addr) = 4 bytes → single TXD0 */
    uint32_t tx = CMD_BLOCK_ERASE_64K |
                  ((addr >> 16) << 8) |
                  ((addr >> 8) << 16) |
                  ((addr & 0xFF) << 24);
    qspi_xfer(tx, 4);

    return flash_wait_ready(3000);
}

/* Program one page (up to 256 bytes) at 3-byte address */
static int flash_program_page(uint32_t addr, const uint8_t *data, int len) {
    int i;
    uint32_t tx[66];  /* max: 4 header bytes + 256 data bytes = 260 → 65 words */
    uint32_t rx[66];

    flash_cmd1(CMD_WRITE_ENABLE);
    delay_us(10);

    /* Verify WEL is set */
    uint8_t sr = flash_read_status();
    if (!(sr & 0x02)) {
        ctrl[8] = sr; /* store raw status for debugging */
        return -2; /* WEL not set */
    }

    /* Build TX buffer: cmd(1) + addr(3) + data(len) */
    uint8_t hdr[4];
    hdr[0] = CMD_PAGE_PROGRAM;
    hdr[1] = (addr >> 16) & 0xFF;
    hdr[2] = (addr >> 8) & 0xFF;
    hdr[3] = addr & 0xFF;

    /* Combine header + data into byte stream, then pack into words */
    int total_bytes = 4 + len;
    int nwords = (total_bytes + 3) / 4;
    if (nwords > 66) return -1;

    /* First word is the header (cmd + 3-byte addr) */
    tx[0] = hdr[0] | (hdr[1] << 8) | (hdr[2] << 16) | (hdr[3] << 24);

    /* Pack data bytes into remaining words */
    for (i = 0; i < len; i += 4) {
        uint32_t w = 0;
        int j;
        for (j = 0; j < 4 && (i + j) < len; j++)
            w |= (uint32_t)data[i + j] << (j * 8);
        /* Pad remaining bytes with 0xFF */
        for (; j < 4; j++)
            w |= 0xFFu << (j * 8);
        tx[1 + i / 4] = w;
    }

    qspi_xfer_multi(tx, rx, nwords);

    return flash_wait_ready(10);
}

/* Read data from flash (3-byte address) */
static void flash_read_data(uint32_t addr, uint8_t *buf, int len) {
    int i;
    /* cmd(1) + addr(3) + data(len) = 4+len bytes */
    int total_bytes = 4 + len;
    int nwords = (total_bytes + 3) / 4;
    uint32_t tx[66], rx[66]; /* max 256 data bytes */
    if (nwords > 66) return;

    /* First word: cmd + 3-byte addr */
    tx[0] = CMD_READ_DATA |
            (((addr >> 16) & 0xFF) << 8) |
            (((addr >> 8) & 0xFF) << 16) |
            ((addr & 0xFF) << 24);

    /* Remaining words: dummy 0xFF to clock in data */
    for (i = 1; i < nwords; i++) tx[i] = 0xFFFFFFFF;

    qspi_xfer_multi(tx, rx, nwords);

    /* Extract data bytes from rx (skip first 4 bytes = cmd+addr echo) */
    uint8_t *rxbytes = (uint8_t *)rx;
    for (i = 0; i < len; i++) {
        buf[i] = rxbytes[4 + i];
    }
}

void _start(void) {
    uint32_t jedec_id;

    /* Initialize QSPI controller */
    qspi_init();

    /* Read JEDEC ID */
    jedec_id = flash_read_jedec();
    ctrl[6] = jedec_id;

    /* Enable Quad SPI mode (needed for BootROM QSPI boot) */
    if (jedec_id != 0) {
        flash_enable_quad();
        /* Store SR2 in ctrl[7] for diagnostic readback */
        ctrl[7] = flash_read_status2();
    }

    /* Signal ready */
    ctrl[3] = 0;
    ctrl[0] = 0;

    /* Main loop */
    while (1) {
        uint32_t cmd = ctrl[0];

        if (cmd == 1) {
            /* === Erase + Program === */
            uint32_t src_addr = ctrl[1];
            uint32_t length = ctrl[2];
            const uint8_t *src = (const uint8_t *)src_addr;

            ctrl[3] = 0;  /* busy */
            ctrl[4] = 0;  /* progress */
            ctrl[5] = 0;  /* no error */

            /* Erase blocks */
            uint32_t erase_end = (length + BLOCK_SIZE - 1) & ~(BLOCK_SIZE - 1);
            uint32_t addr;
            for (addr = 0; addr < erase_end; addr += BLOCK_SIZE) {
                if (flash_erase_block(addr) != 0) {
                    ctrl[3] = 0xFF;
                    ctrl[5] = addr;
                    goto done;
                }
                ctrl[4] = addr;
            }

            /* Program pages */
            uint32_t offset = 0;
            while (offset < length) {
                int chunk = length - offset;
                if (chunk > PAGE_SIZE) chunk = PAGE_SIZE;

                int rc = flash_program_page(offset, &src[offset], chunk);
                if (rc != 0) {
                    ctrl[3] = 0xFF;
                    ctrl[5] = offset;
                    ctrl[7] = (uint32_t)rc; /* error code: -1=timeout, -2=WEL fail */
                    goto done;
                }
                offset += chunk;
                ctrl[4] = offset;
            }

            ctrl[3] = 1;  /* done */
        }
        else if (cmd == 2) {
            /* === Verify === */
            uint32_t src_addr = ctrl[1];
            uint32_t length = ctrl[2];
            const uint8_t *src = (const uint8_t *)src_addr;

            ctrl[3] = 0;
            ctrl[4] = 0;
            ctrl[5] = 0;

            uint8_t buf[256];
            uint32_t offset = 0;
            int ok = 1;

            while (offset < length && ok) {
                int chunk = length - offset;
                if (chunk > 256) chunk = 256;

                flash_read_data(offset, buf, chunk);

                int i;
                for (i = 0; i < chunk; i++) {
                    if (buf[i] != src[offset + i]) {
                        ctrl[5] = offset + i;
                        /* Store debug info: first 4 flash bytes + first 4 DDR bytes */
                        ctrl[7] = buf[0] | (buf[1] << 8) |
                                  (buf[2] << 16) | (buf[3] << 24);
                        ctrl[8] = src[offset] | (src[offset+1] << 8) |
                                  (src[offset+2] << 16) | (src[offset+3] << 24);
                        ctrl[9] = buf[i]; /* actual byte that mismatched */
                        ctrl[10] = src[offset + i]; /* expected byte */
                        ok = 0;
                        break;
                    }
                }
                offset += chunk;
                ctrl[4] = offset;
            }

            ctrl[3] = ok ? 1 : 0xFF;
        }

done:
        ctrl[0] = 0;
        while (ctrl[0] == 0) {
            __asm__ volatile("wfi");
        }
    }
}
