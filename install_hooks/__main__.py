from install_hooks import _install_torch_variant, _install_extras

def main():
    print("[installer] Running SAE-Protein-Design setup...")
    _install_torch_variant()
    _install_extras()
    print("[installer] Environment setup complete!")

if __name__ == "__main__":
    main()
