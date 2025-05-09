import traceback
from train import train

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc() 