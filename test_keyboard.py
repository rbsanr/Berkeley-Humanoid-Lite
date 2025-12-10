

import sys
from inputs import devices

def main():
    """
    This script tests keyboard input using the 'inputs' library.
    It checks for available keyboards and prints any events it receives.
    """
    print("Checking for keyboards...")

    if not devices.keyboards:
        print("Error: No keyboard found. Please make sure a keyboard is connected.")
        sys.exit(1)

    print(f"Found keyboard: {devices.keyboards[0].name}")
    print("Press any key to see the events. Press Ctrl+C to exit.")

    try:
        while True:
            # read() is a blocking call that returns a list of events
            events = devices.keyboards[1].read()
            for event in events:
                print(f"Type: {event.ev_type}, Code: {event.code}, State: {event.state}")

    except PermissionError:
        print("\n---------------------------------------------------------------")
        print("PermissionError: Could not read from the keyboard device.")
        print("Please try running this script with sudo:")
        print(f"    sudo {sys.executable} {sys.argv[0]}")
        print("Alternatively, add your user to the 'input' group and then log out and back in:")
        print("    sudo usermod -a -G input $USER")
        print("---------------------------------------------------------------")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nExiting.")
        sys.exit(0)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

