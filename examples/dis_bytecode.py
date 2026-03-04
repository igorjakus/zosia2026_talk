"""
dis.dis() — podgląd bytecode'u Pythona.
Pokazuje co interpreter robi "pod maską" w prostej pętli.
"""
import dis


def loop():
    s = 0
    for i in range(1000):
        s += i
    return s


def numpy_style():
    """Dla porównania — odpowiednik w NumPy to:
    np.arange(1000).sum()
    Jedna instrukcja C zamiast 5000 instrukcji bytecode.
    """
    import numpy as np
    return np.arange(1000).sum()


if __name__ == "__main__":
    print("=" * 60)
    print("Bytecode pętli Pythona:")
    print("=" * 60)
    dis.dis(loop)
    print("\n" + "=" * 60)
    print("Bytecode NumPy:")
    print("=" * 60)
    dis.dis(numpy_style)
