import os
import sys
import termios
import tty
import shutil

from inputs import basic_input
from inputs import fl_input
from inputs import dp_input

ring_art = """                                                                                 
                 ...                              
              .::::::::..                         
             ::.       ..:::.                     
            .::          ..:-:.                   
            :-:            ..--:                  
           ..-:              .:::                 
           ..:-:              .:::                
            ..::               .::.               
             ..::.             .:::               
              ..--:             .::               
               ..:::.           .-:               
                 ..::..        .::                
                   ...::::::.:::.                 
                      .........                   
                                                  
###   ###   #  #   ##          ##   ####  ###   #  #  ###   
#  #   #    ## #  #  #        #  #  #      #    #  #  #  #  
#  #   #    ## #  #            #    ###    #    #  #  #  #  
###    #    # ##  # ##          #   #      #    #  #  ###   
# #    #    # ##  #  #        #  #  #      #    #  #  #     
#  #  ###   #  #   ###         ##   ####   #     ##   #     
                                                  
"""


# Define colors
RED = "\033[0;31m"
GREEN = "\033[0;32m"
BLUE = "\033[0;34m"
NC = "\033[0m"  # No Color


def clear_screen():
    os.system('clear' if os.name == 'posix' else 'cls')

def get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def print_menu(items, current_index):
    clear_screen()
    print(ring_art)
    print(f"\n{BLUE}Select the type of ring app you want to work with:{NC}\n")
    for i, item in enumerate(items):
        if i == current_index:
            print(f"{GREEN}{item}{NC}")  # Pastel green
        else:
            print(item)

def main():
    items = ["Simple Ring App","Differential Privacy App", "Federated Learning Ring App"]
    function_paths = ["./functions/basic_ring_function.py","./functions/dp_ring_function.py", "./functions/fl_ring_function.py"]
    inputs = [basic_input,dp_input, fl_input]
    current_index = 0

    while True:
        print_menu(items, current_index)
        key = get_key()

        if key == '\x1b':  # ESC
            key = get_key()
            if key == '[':
                key = get_key()
                if key == 'A':  # Up arrow
                    current_index = (current_index - 1) % len(items)
                elif key == 'B':  # Down arrow
                    current_index = (current_index + 1) % len(items)
        elif key == '\r':  # Enter
            selected_item = items[current_index]
            shutil.copy(function_paths[current_index], "ring_function.py")

            print(f"\n{BLUE}Now set the configuration acording to the {items[current_index]}:{NC}\n")
            inputs[current_index].get_inputs()
            break

if __name__ == "__main__":
    main()
