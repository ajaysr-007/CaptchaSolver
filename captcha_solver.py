import sys
from captcha_solver import CaptchaSolver

if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "captcha.png"
    solver = CaptchaSolver()
    result = solver.solve(image_path, verbose=True)
    print("=" * 50)
    print(f"CAPTCHA TEXT: {result}")
    print("=" * 50)
