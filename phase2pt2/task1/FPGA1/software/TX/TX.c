/*
 * Date: 12/10/25
 * Creator: Cesar Pineda
 * Project: Task 1 Transmitter
 */


#include "system.h"
#include "io.h"
#include "alt_types.h"

int main() {
    while (1) {
        // Read the single switch bit (0 or 1)
        alt_u8 sw = IORD_8DIRECT(SWITCH_BASE, 0) & 0x01;

        // Drive the GPIO output pin with the switch value
        IOWR_8DIRECT(GPIO_OUT_BASE, 0, sw);
    }

    return 0;
}
