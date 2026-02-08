#!/usr/bin/env python3
"""Download test fixture audio files from public sources.

These files are public domain recordings from Wikimedia Commons and
similar sources. They are too large to store in git, so this script
downloads them on demand.

Usage:
    python scripts/download_fixtures.py          # download all
    python scripts/download_fixtures.py --check  # verify hashes only
"""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

# Each entry: (filename, sha256, url)
# URLs point to Wikimedia Commons / archive.org / other public domain sources.
# TODO: Fill in the actual URLs for each file.
FIXTURES: list[tuple[str, str, str]] = [
    ("19_16_drum_beat.ogg", "76c059ecf2c833d9a5cb9a9234e3f1c60b462dc019489ab7412e4d487ade768e", ""),
    ("ayub.ogg", "ea0d84b38215541f985ce9449c2ad69829f89204c33c6911170e215936e64749", ""),
    ("bach_minuet.ogg", "eea53daa594fdfc11cbddc8f328a7b4caf471e73191ee800f6fd89c0d59f555e", ""),
    ("bach_siciliana.ogg", "f5f1155df09aca45f56c5d1300c734253ebe665d789ed789ab199c79a8a1c795", ""),
    ("baladi.ogg", "3a1009d50b6ad8880b23ba6f5584c9a39d0a45c7d48b821acc8e227f6a8f23ff", ""),
    ("barcarolle_chopin.ogg", "e4344ca7500e598d8624e4a871c0b4fa903ea7b2674a3b603b09ea47f06bc86b", ""),
    ("barcarolle_offenbach.ogg", "7741f9f0f88ec102ac3ac1877d260f60c53f788274f2e8cad229ba5bdc5b1e5d", ""),
    ("blast_beat.ogg", "2601435e7077cc7e9a8880137eb8c954bdd8b20e75db5f6e1591e8af5e2b3ccd", ""),
    ("blowing_bubbles_waltz.ogg", "e6af274ace991f138919ef0e0c1c940aa4427704d2f1cbd9e94e5fdc39ba4ed0", ""),
    ("blue_danube.ogg", "1124cd59575947d9adaaf731f3e2071d412049e6be443bd2d546f25b33007ce3", ""),
    ("blues_guitar.ogg", "35f8908c1e139fba3e95772e5ef72e8b516600a94fb645acea325c55b37e5cb3", ""),
    ("bossa_nova.ogg", "9fc3866ef8aa0a222a238e28bd8394e3408db42040d86c090788b13c0abb4afa", ""),
    ("bushwick_tarantella.oga", "18dd8807e1c767f60541295e192d333966cac76c53753b686b857a2e546ae4eb", ""),
    ("celebre_tarantella.ogg", "bef7d6428f92e1789e090c7b712c5153285292513aeea38d72e21596c757a055", ""),
    ("chopin_waltz.ogg", "3baf37dc60162f5654e9748504a04a0169a2047663a64cda5d1960a64fc44451", ""),
    ("djembe.ogg", "ab96dafbf7e6bd43072d514730331007f4fb591668f1e8af30404ecb8a1f0e14", ""),
    ("drum_beat.ogg", "b889f17b89d62f106c13be1f155a70c793fc9ea17259aca60085ffb486ad4edb", ""),
    ("drum_cadence_a.ogg", "5083f07730c6ffb1cd2c3b5cc1a43714a0cfda04fe913935e83d1048466562e6", ""),
    ("drum_cadence_b.ogg", "1486e49ba5a011c9f8127d81d0e2f8aefc7ca7df8f726c03aa6e65002914adfc", ""),
    ("dubstep_drums.ogg", "07bbdf3460ba0cd97bd6af62c6d84d8312d8209c5bb023c5b0a920cb129948f3", ""),
    ("erika_march.ogg", "7affa89eb6a0c38beacb7a673987c7ba3da9d38b82fcc5571c49aad6f5026ed7", ""),
    ("greensleeves.ogg", "90a8fa77b72a314123b42b36bd4b1392050fd218ad439c5377f26fd6153fb59f", ""),
    ("irish_reel_mountain_road.ogg", "ef1c5161ffca7d7ab54dbb79a16736c3832addb7635c63cc20441b12ee8e04bc", ""),
    ("jazz_ride.ogg", "abed5053e9e2a7a9883ecd82ad7e692064812edb45d67a58d427d73efb66c8fc", ""),
    ("jig_doethion.ogg", "846f9dcaa3afee2ee01441c52c47001ef3a9f8a0e85f461e1c48a87384242390", ""),
    ("jigs_gwerinos.ogg", "52c8c5c56d78a0baa5839e18e90e7732ecd1ed1fa58074fabe6899f2e8a45dd3", ""),
    ("joplin_waltz.ogg", "1383e4bf8ebb4938534bd2ba07c1af557a00c67a3be12a017a6a6bda61be856e", ""),
    ("lost_train_blues.ogg", "63c4afa65a89d2caf5bf7db4e188ce34ac0d1e5b308906b8e058cdfae543a67b", ""),
    ("maksum.ogg", "e089c14bf072b05da8e21b76bebfab64fcbe52960a58f60a3a35862acd117911", ""),
    ("malfuf.ogg", "9587b43add4778122b8e2831ba9a4684bd9cf795f613996bee2b7df594d2816a", ""),
    ("march_grandioso.ogg", "fe1a4963cd10c58b11fd304af0f5049f3bcef77d8a86446ee25c7c398ea60764", ""),
    ("march_military.ogg", "51be8fdb6a6a528211d6567e7c125b89fb3c79372928bc223479c8a3e9cfaa30", ""),
    ("march_suffrage.ogg", "261775605e8d1312119841563acba181ec19831010122751b8bdc3f9b63a832c", ""),
    ("mazurka_chopin_op7.ogg", "c17d934a1cf77a10abf3b404fdcab61d712510f3bd9b91a302ebeec87fbb8656", ""),
    ("midnight_waltz.ogg", "11548a979bd73bc72b51591f5af4bc32d467ac84b902d5b4949d9c18adfd42f2", ""),
    ("minuet_beethoven.ogg", "8147dc55f9ac85bb742cc56f1a27ead6f6fe637eaf09c5d1d591f3b3ac45b8eb", ""),
    ("minuet_paderewski.ogg", "701174e180755d0b059bf2f8eb6fedce5603df5e4169889ddf6854c0a836af81", ""),
    ("polka_kathi.ogg", "2ca07808d1e328ec7acf8157c5cbea42e3fd68322d288ac53425de759b0981ba", ""),
    ("polka_pixel_peeker.ogg", "3536e96baaf17846929085a4cecb40d553a83ed5d2623cf26495e197fb21f089", ""),
    ("polka_smetana.ogg", "62d50778fa5ba7e453e81b61fdf1d199372d0015ef74fe34b214889ca1f63fd0", ""),
    ("polka_tritsch_tratsch.ogg", "d7bdeb020c8daad2778033f39849f64d8c7a3f483377dc2e618634b36a971a6d", ""),
    ("reggae_one_drop.ogg", "a289e5edfac351bfee6dc1c1687dede31bffa637f7b1365cc2476e9b39234550", ""),
    ("rock_beat.ogg", "7bed4bd6a7e16236fdea1bfc0f34785ba0209b5d1a89fab455ada19b0a995ec5", ""),
    ("roxys_birthday_jigs.ogg", "20e5724ca73a6c590ba96ec89bd73041299e93ff5e5fe97e6ba49bf184a4987b", ""),
    ("saidi.ogg", "5db6a6d994e26d30669a6d440dc42a90fdb67b4e4d6a066e24d0f1f140654151", ""),
    ("sarabande_bach.ogg", "a6c2a4a72b80099d17ce6854fca72522bb4604f6b315260bd0f7fb50410ca304", ""),
    ("sarabande_handel.oga", "1de70fb16ebe05565b4069aac373758f73a6c51c81a8c3c1994c61aa7d69c423", ""),
    ("shuffle.ogg", "e2656d7601d34e0f5917edeb6d0db36c788bed74823ec0a843008bae1d5407ec", ""),
    ("slip_jigs.ogg", "82bb25a7b2d4915d7cde47b145c9c8a16f310d239c61a4680a0138970750b794", ""),
    ("tango_albeniz.ogg", "54226d9de2cec6df0fff5f6d22c2ff71e51fc40f94c64babc3a2d17fc33e0893", ""),
    ("tango_argentino.ogg", "d0bcf937f5cb182cf963c2e6b7d146e0286b182416b8914a525cfad90d68c81c", ""),
    ("tarantella_choir.ogg", "c557ec61bddd03559e76563c1314f062f3e5d7842e8b8e0e0dc7049b6a76c630", ""),
    ("tarantella_napoletana.ogg", "6091d2e2abecba2002995c26dd2469377693f29604d5eaaca87b43df7a519c0a", ""),
    ("tarantella_welsh_tenors.ogg", "b5da8708dd5cc9f073be47ec0bd3c16b4be565aed9dd82d331716f8e5b47df58", ""),
    ("waltz_stefan.ogg", "0ecb742cc963983424faf69e414e619f822f44f70f1d72e185d4a975e1a382db", ""),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path) -> bool:
    try:
        print(f"  Downloading {dest.name}...")
        urllib.request.urlretrieve(url, str(dest))
        return True
    except Exception as e:
        print(f"  ERROR downloading {dest.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download test fixture audio files")
    parser.add_argument("--check", action="store_true",
                        help="Only verify hashes of existing files")
    args = parser.parse_args()

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    ok = 0
    missing = 0
    failed = 0
    no_url = 0

    for filename, expected_hash, url in FIXTURES:
        dest = FIXTURES_DIR / filename

        if args.check:
            if not dest.exists():
                print(f"  MISSING: {filename}")
                missing += 1
                continue
            actual = sha256_file(dest)
            if actual == expected_hash:
                ok += 1
            else:
                print(f"  HASH MISMATCH: {filename}")
                print(f"    expected: {expected_hash}")
                print(f"    actual:   {actual}")
                failed += 1
            continue

        # Download mode
        if dest.exists():
            actual = sha256_file(dest)
            if actual == expected_hash:
                ok += 1
                continue
            else:
                print(f"  Hash mismatch for {filename}, re-downloading...")

        if not url:
            print(f"  SKIP: {filename} (no URL configured)")
            no_url += 1
            continue

        if not download_file(url, dest):
            failed += 1
            continue

        actual = sha256_file(dest)
        if actual != expected_hash:
            print(f"  HASH MISMATCH after download: {filename}")
            print(f"    expected: {expected_hash}")
            print(f"    actual:   {actual}")
            dest.unlink()
            failed += 1
        else:
            ok += 1

    print(f"\nResults: {ok} ok, {missing} missing, {failed} failed, {no_url} no URL")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
