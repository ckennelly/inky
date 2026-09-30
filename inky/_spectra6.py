"""Palette handling shared by the Spectra 6 drivers (E640, E673, EL133UF1)."""
import numpy


def check_palette(palette):
    """Six (r, g, b) colours as a list of tuples, or ValueError."""
    try:
        colours = [tuple(int(c) for c in rgb) for rgb in palette]
    except (TypeError, ValueError):
        raise ValueError("palette must be six (r, g, b) colours") from None
    if len(colours) != 6 or any(len(c) != 3 or not all(0 <= v <= 255 for v in c)
                                for c in colours):
        raise ValueError("palette must be six (r, g, b) colours with values 0-255")
    return colours


def _lab(rgb):
    """sRGB 0-255 to CIELAB (D65)."""
    a = numpy.asarray(rgb, dtype=numpy.float64).reshape(-1, 3) / 255.0
    lin = numpy.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    to_xyz = numpy.array([[0.4124, 0.3576, 0.1805],
                          [0.2126, 0.7152, 0.0722],
                          [0.0193, 0.1192, 0.9505]])
    xyz = lin @ to_xyz.T / numpy.array([0.95047, 1.0, 1.08883])
    f = numpy.where(xyz > 0.008856, numpy.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return numpy.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], axis=1)


def _inks_by_hue(colours, saturated, chroma_floor=15.0):
    """Map each colour to an ink by what kind of colour it is, not how close.

    For flat-colour images -- dashboards, charts, pixel art -- the useful
    question is "which ink is this the colour of", not "which ink is it
    nearest to in RGB": nearest makes light blue white and dark yellow
    olive-black. Colours with little chroma are black or white by
    lightness; the rest take the ink with the nearest hue angle, using the
    hues of the inks as they really look (SATURATED_PALETTE).
    """
    lab = _lab(colours)
    chroma = numpy.hypot(lab[:, 1], lab[:, 2])
    hue = numpy.arctan2(lab[:, 2], lab[:, 1])
    ink = _lab(saturated[:6])
    ink_hue = numpy.arctan2(ink[2:6, 2], ink[2:6, 1])  # yellow, red, blue, green
    apart = numpy.abs(numpy.angle(numpy.exp(1j * (hue[:, None] - ink_hue[None, :]))))
    return numpy.where(chroma < chroma_floor,
                       numpy.where(lab[:, 0] < 50, 0, 1),
                       2 + apart.argmin(axis=1))


def _nearest_inks(colours, saturated, desaturated, extra=None):
    """Index of the nearest ink for each RGB colour.

    To these drivers an ink is not a single colour: set_image() dithers
    against a blend of its pure (DESATURATED) and realistic (SATURATED)
    colour, anywhere between the two depending on `saturation`. So measure
    each colour's distance to that whole segment rather than to its pure
    end. Every palette the driver itself can dither with then maps back to
    the right inks by construction -- and the dark saturated blue and green
    no longer lose to black, which is nearer than pure blue or pure green.
    """
    colours = numpy.asarray(colours, dtype=numpy.float64).reshape(-1, 3)
    distance = numpy.empty((len(colours), 6))
    for ink in range(6):
        a = numpy.asarray(desaturated[ink], dtype=numpy.float64)
        ab = numpy.asarray(saturated[ink], dtype=numpy.float64) - a
        length2 = float(ab @ ab)
        # Black is the same in both palettes, so its segment is a point.
        t = numpy.zeros(len(colours)) if length2 == 0.0 else numpy.clip((colours - a) @ ab / length2, 0.0, 1.0)
        distance[:, ink] = numpy.linalg.norm(colours - (a + t[:, None] * ab), axis=1)
        if extra is not None:
            # A caller's own palette is another place this ink can be.
            point = numpy.linalg.norm(colours - numpy.asarray(extra[ink], dtype=numpy.float64), axis=1)
            distance[:, ink] = numpy.minimum(distance[:, ink], point)
    return distance.argmin(axis=1)


def palette_image_to_inks(image, saturated, desaturated, extra=None, match="rgb"):
    """Map a palette-mode image straight to ink indices, if it is one.

    An image using at most six palette indices is taken to be already
    dithered for the display, so each index is mapped to an ink and its
    pixels are left exactly where the caller put them. Re-quantizing the
    pixel colours instead, as set_image() used to, both re-dithers the
    image and loses blue and green whenever the palette holds their
    realistic colours (see pimoroni/inky#221).

    Only indices that pixels actually use are counted, so padding a palette
    out to 256 entries does not turn an unused (0, 0, 0) into a seventh
    colour.

    `extra`, the caller's own palette if it passed one, counts as a further
    colour for each ink, so an image labelled with that palette round-trips
    even where its colours would be ambiguous against the driver's own.

    With match="hue" every palette entry is mapped by hue instead, however
    many there are, and nothing is dithered -- for flat-colour images.

    Returns ink indices (0-5) shaped like the image, or None when the image
    uses more than six colours, or colours it has no palette entry for, and
    so should be dithered like any other image.
    """
    used = image.getcolors(256 if match == "hue" else 6)
    if used is None:
        return None
    used = numpy.array([index for _, index in used])
    lut = numpy.zeros(256, dtype=numpy.uint8)

    if not image.palette.colors:
        # No colour information at all: take indices as inks in their
        # default order, as set_image() always has.
        if used.max() > 5:
            return None
        lut[:6] = numpy.arange(6)
    else:
        flat = image.getpalette() or []
        if len(flat) < 3 * (int(used.max()) + 1):
            return None
        colours = [flat[3 * i:3 * i + 3] for i in used]
        if match == "hue":
            lut[used] = _inks_by_hue(colours, saturated)
        else:
            lut[used] = _nearest_inks(colours, saturated, desaturated, extra)

    return lut[numpy.asarray(image, dtype=numpy.uint8)]
