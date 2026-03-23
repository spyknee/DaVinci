"""Quick smoke test for DaVinci v0.5.1 — safe for double-click."""
import sys, os

# Ensure local davinci/ is importable
sys.path.insert(0, os.getcwd())

try:
    from davinci.interface.api import DaVinci
    from davinci.fractals import escape_time, normalize_escape_time

    print("\n🧪 Starting DaVinci v0.5.1 tests...\n")

    # Test 1: fractal math
    z0 = complex(0.5, 0.5)
    c  = complex(-0.7269, 0.1889)  # connected-Julia parameter (Stella Artois point 😄)
    t_esc = escape_time(z0, c)
    norm = normalize_escape_time(t_esc)
    
    print(f"Fractal math test:")
    print(f"  z₀ = {z0}, c = {c}")
    print(f"  → escape time = {t_esc:.2f} (smoothed iterations)")
    print(f"  → normalized retention potential = {norm:.3f}")
    print("✅ Fractal math works!\n")

    # Test 2: memory lifecycle
    db_path = "davinci_test.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    with DaVinci(db_path=db_path) as dv:
        mid = dv.remember("The sky is blue", zoom_levels={1: "Weather observation"})
        print(f"✅ Stored memory ID: {mid[:8]}...")

        node = dv.recall(mid)
        assert node and node.content == "The sky is blue"
        print("✅ Recalled correctly ✅\n")

        changed = dv.decay()
        print(f"✅ Decay cycle completed — reclassified memories: {changed}\n")

    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)

    print("=" * 40)
    print("✨ ALL TESTS PASSED ✨")
    print("=" * 40)

except Exception as e:
    print(f"\n❌ Error during test:\n{type(e).__name__}: {e}\n")
    import traceback
    traceback.print_exc()

# Pause before closing (Windows/macOS double-click friendly)
input("\n✅ Press Enter to close... ")
