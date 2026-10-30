"""Parsing dos adapters reais com respostas simuladas (offline) + testes de rede marcados."""
import io

import httpx
import numpy as np
import pytest
from astropy.io import fits

from spherex.infrastructure.allwise_client import AllWiseCatalog
from spherex.infrastructure.irsa_client import IrsaSpherexSource
from spherex.infrastructure.jpl_clients import HorizonsEphemeris
from spherex.pipeline.geometry import make_wcs_header

VOTABLE = b"""<?xml version="1.0"?>
<VOTABLE version="1.4" xmlns="http://www.ivoa.net/xml/VOTable/v1.3"><RESOURCE type="results"><TABLE>
<FIELD name="obs_publisher_did" datatype="char" arraysize="*"/>
<FIELD name="access_url" datatype="char" arraysize="*"/>
<FIELD name="t_min" datatype="double"/><FIELD name="em_min" datatype="double"/><FIELD name="em_max" datatype="double"/>
<DATA><TABLEDATA>
<TR><TD>ivo://irsa/spherex/b</TD><TD>https://example.org/b.fits</TD><TD>61001.5</TD><TD>1.10e-6</TD><TD>1.14e-6</TD></TR>
<TR><TD>ivo://irsa/spherex/a</TD><TD>https://example.org/a.fits</TD><TD>61000.5</TD><TD>1.08e-6</TD><TD>1.12e-6</TD></TR>
</TABLEDATA></DATA></TABLE></RESOURCE></VOTABLE>"""


def _fits_bytes() -> bytes:
    hdr = fits.Header()
    for k, v in make_wcs_header(150.0, 2.0, (16, 16), 6.15).items():
        if not k.startswith("NAXIS"):
            hdr[k] = v
    buf = io.BytesIO()
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.ones((16, 16), np.float32), hdr, name="IMAGE"),
                  fits.ImageHDU(np.zeros((16, 16), np.uint8), name="FLAGS")]).writeto(buf)
    return buf.getvalue()


def test_irsa_sia_parsing_and_download():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/SIA":
            seen["pos"] = req.url.params["POS"]
            seen["maxrec"] = req.url.params["MAXREC"]
            return httpx.Response(200, content=VOTABLE)
        seen["cutout"] = dict(req.url.params)
        return httpx.Response(200, content=_fits_bytes())

    src = IrsaSpherexSource(client=httpx.Client(transport=httpx.MockTransport(handler)), min_interval_s=0)
    ids = src.list_exposures("f1", 150.0, 2.0, 0.1)
    assert ids == ["ivo://irsa/spherex/a", "ivo://irsa/spherex/b"]          # ordenado por época
    assert seen["pos"] == "circle 150.0 2.0 0.1"          # sintaxe dos exemplos oficiais do IRSA
    assert seen["maxrec"] == "2000"
    exp = src.load(ids[0])
    # sem WCS-WAVE no arquivo: cai na média em_min/em_max do SIA (metros -> µm)
    assert abs(exp.wavelength_um - 1.10) < 1e-9 and exp.image.shape == (16, 16)
    assert exp.mask is not None
    assert seen["cutout"] == {"center": "150.0,2.0", "size": "0.1", "gzip": "false"}


def test_irsa_band_filter_in_meters():
    seen = {}

    def handler(req):
        seen.update(req.url.params)
        return httpx.Response(200, content=VOTABLE)

    src = IrsaSpherexSource(client=httpx.Client(transport=httpx.MockTransport(handler)), min_interval_s=0,
                            band_um=(1.10, 1.14))
    src.list_exposures("f", 150.0, 2.0, 0.1)
    assert seen["BAND"] == "1.100000e-06 1.140000e-06"


def test_flag_bitmask_selects_only_requested_bits():
    def handler(req):
        if req.url.path == "/SIA":
            return httpx.Response(200, content=VOTABLE)
        hdr = fits.Header()
        for k, v in make_wcs_header(150.0, 2.0, (4, 4), 6.15).items():
            if not k.startswith("NAXIS"):
                hdr[k] = v
        flags = np.array([[0, 1, 2, 4]] * 4, np.int32)
        buf = io.BytesIO()
        fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.ones((4, 4), np.float32), hdr, name="IMAGE"),
                      fits.ImageHDU(flags, name="FLAGS")]).writeto(buf)
        return httpx.Response(200, content=buf.getvalue())

    src = IrsaSpherexSource(client=httpx.Client(transport=httpx.MockTransport(handler)), min_interval_s=0,
                            flag_bits=0b100)
    exp = src.load(src.list_exposures("f", 150.0, 2.0, 0.1)[0])
    assert exp.mask[0].tolist() == [False, False, False, True]


def test_sip_distortion_survives_cache_roundtrip(tmp_path):
    from spherex.domain.models import Exposure
    from spherex.infrastructure.fits_io import celestial_header, read_exposure, write_exposure
    from spherex.pipeline.geometry import pixel_to_world
    h = fits.Header()
    for k, v in make_wcs_header(150.0, 2.0, (64, 64), 6.15).items():
        h[k] = v
    h["CTYPE1"], h["CTYPE2"] = "RA---TAN-SIP", "DEC--TAN-SIP"
    h["A_ORDER"], h["B_ORDER"], h["A_2_0"], h["B_0_2"] = 2, 2, 2e-4, -3e-4
    wh = celestial_header(h)
    assert wh["A_2_0"] == 2e-4 and wh["B_0_2"] == -3e-4
    exp = Exposure("e", "f", 61000.0, "D1", 1.1, np.zeros((64, 64), np.float32), wh)
    write_exposure(exp, tmp_path / "e.fits")
    back = read_exposure(tmp_path / "e.fits")
    ra1, dec1 = pixel_to_world(wh, [2.0], [60.0])
    ra2, dec2 = pixel_to_world(back.wcs_header, [2.0], [60.0])
    assert abs(ra1[0] - ra2[0]) < 1e-10 and abs(dec1[0] - dec2[0]) < 1e-10


def test_allwise_csv_parsing():
    def handler(req):
        assert "allwise_p3as_psd" in req.url.params["QUERY"]
        return httpx.Response(200, text="ra,dec\n150.1,2.1\n150.2,1.9\n")

    cat = AllWiseCatalog(client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert cat.sources(150.0, 2.0, 0.2) == [(150.1, 2.1), (150.2, 1.9)]


def test_horizons_vector_parsing():
    result = ("header\n$$SOE\n2461000.500000000, A.D. 2026-Jan-01 00:00:00.0000, "
              "-1.234567890123456E-01, 9.876543210987654E-01, 1.000000000000000E-04, "
              "-1.700000000000000E-02, -2.100000000000000E-03, 1.000000000000000E-06,\n$$EOE\n")

    def handler(req):
        assert req.url.params["COMMAND"] == "'-163182'"
        return httpx.Response(200, json={"result": result})

    eph = HorizonsEphemeris(client=httpx.Client(transport=httpx.MockTransport(handler)))
    x, y, z = eph.observer_position(61000.0)
    assert abs(x + 0.1234567890123456) < 1e-12 and abs(y - 0.9876543210987654) < 1e-12


@pytest.mark.network
def test_real_irsa_sia_returns_spherex_exposures():
    ids = IrsaSpherexSource().list_exposures("cosmos", 150.1, 2.2, 0.05)
    assert ids, "SIA não retornou exposições: confira COLLECTION e POS em docs/dados.md"


@pytest.mark.network
def test_real_horizons_spherex_position():
    x, y, z = HorizonsEphemeris().observer_position(61000.0)
    assert 0.9 < (x * x + y * y + z * z) ** 0.5 < 1.1   # órbita terrestre baixa: ~1 au do Sol
