/*
 * Date: 12/10/25
 * Creator: Cesar Pineda
 * Project: Task 1 Receiver
 */

#include "system.h"
#include "io.h"
#include "alt_types.h"
#include "altera_avalon_pio_regs.h"

int main() {
    while (1) {
        // Read 1-bit GPIO input from FPGA1
        alt_u8 in_bit = IORD_ALTERA_AVALON_PIO_DATA(GPIO_IN_BASE) & 0x01;

        // Write that bit to LED[0]
        IOWR_ALTERA_AVALON_PIO_DATA(LED_BASE, in_bit);
    }

    return 0;
}
