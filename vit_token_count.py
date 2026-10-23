RESOLUTIONS = [224, 384, 512, 1024]
PATCH_SIZES = [8, 14, 16, 32]
NUM_HEADS = 12
BASE_TOKENS = (224 // 16) ** 2 + 1


def main():
    print(f"{'Resolution':>10} {'Patch':>5} {'Grid':>7} {'N':>6} {'N+1':>6} "
          f"{'Attn, MB':>9} {'vs 224/16':>9}")
    for res in RESOLUTIONS:
        for p in PATCH_SIZES:
            grid = res // p
            n = grid * grid
            tokens = n + 1
            attn_mb = NUM_HEADS * tokens ** 2 * 4 / 1e6     # float32, one layer, B=1
            ratio = (tokens / BASE_TOKENS) ** 2
            print(f"{res:>10} {p:>5} {f'{grid}x{grid}':>7} {n:>6} {tokens:>6} "
                  f"{attn_mb:>9.1f} {ratio:>8.1f}x")


if __name__ == "__main__":
    main()
