"""set_image tests for the Spectra 6 displays (E640, E673, EL133UF1).

A palette-mode image is how a caller hands the driver a picture it has
already dithered itself. These tests check that each palette index lands
on the ink it names, whichever of the driver's own palettes -- pure,
saturated, or a blend of the two -- the caller labelled the image with.
"""
import importlib

import pytest

DRIVERS = [
    ("inky.inky_e640", (600, 400)),
    ("inky.inky_e673", (800, 480)),
    ("inky.inky_el133uf1", (1600, 1200)),
]

# set_image() stores the panel's native codes, which skip 4.
NATIVE = [0, 1, 2, 3, 5, 6]
INKS = ["black", "white", "yellow", "red", "blue", "green"]


def make_display(name, GPIO, spidev, smbus2):
    module = importlib.import_module(name)
    return module, module.Inky()


def banded(size, palette6):
    """A P image in six vertical bands, band i holding palette index i."""
    from PIL import Image

    width, height = size
    image = Image.new("P", size)
    image.putpalette([c for rgb in palette6 for c in rgb])
    row = [(x * 6) // width for x in range(width)]
    image.putdata(row * height)
    return image


def inks_per_band(display, size):
    """The set of native codes set_image stored in each band."""
    width, _ = size
    buf = display.buf.reshape(-1, width)
    return [set(buf[:, (i * width) // 6 + 1:((i + 1) * width) // 6 - 1].ravel().tolist())
            for i in range(6)]


def assert_bands(display, size, expected):
    got = inks_per_band(display, size)
    for i, (band, want) in enumerate(zip(got, expected)):
        assert band == {NATIVE[want]}, (
            f"band {i} should be {INKS[want]} (native {NATIVE[want]}), got native {sorted(band)}")


@pytest.mark.parametrize("name,size", DRIVERS)
def test_pure_palette_keeps_every_ink(GPIO, spidev, smbus2, name, size):
    module, display = make_display(name, GPIO, spidev, smbus2)
    display.set_image(banded(size, [tuple(c) for c in module.DESATURATED_PALETTE[:6]]))
    assert_bands(display, size, range(6))


@pytest.mark.parametrize("name,size", DRIVERS)
def test_saturated_palette_keeps_every_ink(GPIO, spidev, smbus2, name, size):
    """Labelling with the driver's own SATURATED_PALETTE must not lose inks.

    Blue (61, 59, 94) and green (58, 91, 70) are dark, so matching them
    against pure primaries finds black nearer than blue or green, and
    both bands used to reach the panel as black.
    """
    module, display = make_display(name, GPIO, spidev, smbus2)
    display.set_image(banded(size, [tuple(c) for c in module.SATURATED_PALETTE[:6]]))
    assert_bands(display, size, range(6))


@pytest.mark.parametrize("saturation", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("name,size", DRIVERS)
def test_blended_palette_keeps_every_ink(GPIO, spidev, smbus2, name, size, saturation):
    """Any palette the driver itself would dither with must round-trip."""
    _module, display = make_display(name, GPIO, spidev, smbus2)
    flat = display._palette_blend(saturation)
    display.set_image(banded(size, [tuple(flat[i:i + 3]) for i in range(0, 18, 3)]))
    assert_bands(display, size, range(6))


@pytest.mark.parametrize("name,size", DRIVERS)
def test_reordered_palette_maps_by_colour(GPIO, spidev, smbus2, name, size):
    """Six correct colours in some other order still land on the right inks."""
    module, display = make_display(name, GPIO, spidev, smbus2)
    order = [3, 0, 5, 1, 4, 2]
    palette = [tuple(module.SATURATED_PALETTE[i]) for i in order]
    display.set_image(banded(size, palette))
    assert_bands(display, size, order)


@pytest.mark.parametrize("name,size", DRIVERS)
def test_rgb_image_of_pure_colours_uses_every_ink(GPIO, spidev, smbus2, name, size):
    module, display = make_display(name, GPIO, spidev, smbus2)
    image = banded(size, [tuple(c) for c in module.DESATURATED_PALETTE[:6]]).convert("RGB")
    display.set_image(image)
    assert_bands(display, size, range(6))


@pytest.mark.parametrize("name,size", DRIVERS)
def test_predithered_image_is_not_redithered(GPIO, spidev, smbus2, name, size):
    """An image already dithered to the six inks must pass through untouched.

    Random indices stand in for any caller's own dithering. Every pixel
    must arrive as the ink its index names: re-quantizing would move
    pixels, and error diffusion would move neighbours too.
    """
    import numpy
    from PIL import Image

    module, display = make_display(name, GPIO, spidev, smbus2)
    width, height = size
    indices = numpy.random.default_rng(221).integers(0, 6, (height, width), dtype=numpy.uint8)
    image = Image.fromarray(indices, "P")
    image.putpalette([c for rgb in module.SATURATED_PALETTE[:6] for c in rgb])
    display.set_image(image)
    expected = numpy.array(NATIVE, dtype=numpy.uint8)[indices]
    assert (display.buf.reshape(height, width) == expected).all()


@pytest.mark.parametrize("name,size", DRIVERS)
def test_padded_palette_with_non_black_first_entry(GPIO, spidev, smbus2, name, size):
    """Unused padding entries must not count as colours.

    If the palette's black is not exactly (0, 0, 0), padding it to 256
    entries with zeros makes pure black a seventh palette colour even
    though no pixel uses it.
    """
    module, display = make_display(name, GPIO, spidev, smbus2)
    palette = [(24, 24, 26)] + [tuple(c) for c in module.SATURATED_PALETTE[1:6]]
    image = banded(size, palette)
    image.putpalette([c for rgb in palette for c in rgb] + [0, 0, 0] * 250)
    display.set_image(image)
    assert_bands(display, size, range(6))


@pytest.mark.parametrize("name,size", DRIVERS)
def test_p_image_without_palette_uses_index_order(GPIO, spidev, smbus2, name, size):
    """Indices with no palette at all are inks in the default order."""
    from PIL import Image

    _module, display = make_display(name, GPIO, spidev, smbus2)
    width, height = size
    image = Image.new("P", size)
    image.putdata([(x * 6) // width for x in range(width)] * height)
    assert not image.palette.colors
    display.set_image(image)
    assert_bands(display, size, range(6))


# --- palette= ----------------------------------------------------------------

MEASURED = [(57, 42, 59), (254, 254, 253), (254, 254, 0),
            (213, 132, 21), (68, 90, 204), (112, 149, 123)]


def random_rgb(size, seed=0):
    import numpy
    from PIL import Image

    width, height = size
    rng = numpy.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=numpy.uint8), "RGB")


@pytest.mark.parametrize("name,size", DRIVERS)
def test_default_palette_is_unchanged(GPIO, spidev, smbus2, name, size):
    """palette=None must dither exactly as before: same buffer, bit for bit."""
    _module, display = make_display(name, GPIO, spidev, smbus2)
    image = random_rgb(size)
    display.set_image(image, saturation=0.5)
    before = display.buf.copy()
    display.set_image(image, saturation=0.5, palette=None)
    assert (display.buf == before).all()


@pytest.mark.parametrize("name,size", DRIVERS)
def test_custom_palette_dithers_rgb_against_it(GPIO, spidev, smbus2, name, size):
    """Pure colours still land on their inks when dithered against a custom palette."""
    module, display = make_display(name, GPIO, spidev, smbus2)
    image = banded(size, [tuple(c) for c in module.DESATURATED_PALETTE[:6]]).convert("RGB")
    display.set_image(image, palette=MEASURED)
    assert_bands(display, size, range(6))


@pytest.mark.parametrize("name,size", DRIVERS)
def test_custom_palette_changes_the_dither(GPIO, spidev, smbus2, name, size):
    """A different target palette must actually change the result."""
    _module, display = make_display(name, GPIO, spidev, smbus2)
    image = random_rgb(size, seed=1)
    display.set_image(image)
    default = display.buf.copy()
    display.set_image(image, palette=MEASURED)
    assert (display.buf != default).mean() > 0.05


@pytest.mark.parametrize("name,size", DRIVERS)
def test_p_image_labelled_with_custom_palette_round_trips(GPIO, spidev, smbus2, name, size):
    """Labelled with the palette also passed as palette=, every ink survives.

    This palette's black (57, 42, 59) is nearer the driver's saturated blue
    than its black, so without palette= to say otherwise it is ambiguous.
    """
    _module, display = make_display(name, GPIO, spidev, smbus2)
    display.set_image(banded(size, MEASURED), palette=MEASURED)
    assert_bands(display, size, range(6))


@pytest.mark.parametrize("bad", [[(0, 0, 0)] * 5, [(0, 0, 0)] * 7, [(0, 0)] * 6])
def test_palette_must_be_six_rgb_colours(GPIO, spidev, smbus2, bad):
    _module, display = make_display("inky.inky_el133uf1", GPIO, spidev, smbus2)
    with pytest.raises(ValueError):
        display.set_image(random_rgb((1600, 1200)), palette=bad)
