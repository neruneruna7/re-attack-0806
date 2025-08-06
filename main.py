import torch
def main():
    # pythochで，metalGPUが使えているか確認する
    if torch.backends.mps.is_available():
        print("Metal GPU is available.")
    else:
        print("Metal GPU is not available.")
    print("Hello from re-attack-0806!")


if __name__ == "__main__":
    main()
