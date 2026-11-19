//! The Windows icon resource, so Explorer, the taskbar and Alt+Tab show the mark on `litmus-app.exe` itself. The
//! build script (`build.rs`) draws the mark with `brand::icon_rgba` at the usual icon sizes and writes them as a
//! compiled resource file (`.res`: an icon group and its images), which the MSVC linker takes as an input like an
//! object file; no resource compiler is needed. Small sizes are stored as 32-bit bitmaps, 256 as a PNG.

/// The sizes drawn as bitmaps; 256 is added as a PNG.
pub const SIZES: [usize; 7] = [16, 20, 24, 32, 40, 48, 64];
const RT_ICON: u16 = 3;
const RT_GROUP_ICON: u16 = 14;
const LANGUAGE: u16 = 0x0409;

/// An icon image as Windows stores it: a 32-bit bottom-up bitmap with a doubled height and an empty AND mask.
pub fn dib(size: usize, rgba: &[u8]) -> Vec<u8> {
    assert_eq!(rgba.len(), size * size * 4, "RGBA pixels of the given size");
    let mask_row = size.div_ceil(32) * 4;
    let image = size * size * 4 + mask_row * size;
    let mut out = Vec::with_capacity(40 + image);
    out.extend(40u32.to_le_bytes());
    out.extend((size as i32).to_le_bytes());
    out.extend((2 * size as i32).to_le_bytes());
    out.extend(1u16.to_le_bytes());
    out.extend(32u16.to_le_bytes());
    out.extend(0u32.to_le_bytes()); // no compression
    out.extend((image as u32).to_le_bytes());
    out.extend([0u8; 16]); // resolution and palette: unused
    for row in (0..size).rev() {
        for pixel in rgba[row * size * 4..(row + 1) * size * 4].as_chunks::<4>().0 {
            out.extend([pixel[2], pixel[1], pixel[0], pixel[3]]);
        }
    }
    out.resize(out.len() + mask_row * size, 0);
    out
}

/// The images of the icon, smallest first: (size, bytes).
pub fn images(draw: impl Fn(usize) -> Vec<u8>, png: impl Fn(usize, &[u8]) -> Vec<u8>) -> Vec<(usize, Vec<u8>)> {
    let mut out: Vec<(usize, Vec<u8>)> = SIZES.iter().map(|&size| (size, dib(size, &draw(size)))).collect();
    out.push((256, png(256, &draw(256))));
    out
}

fn entry(out: &mut Vec<u8>, kind: u16, name: u16, flags: u16, data: &[u8]) {
    out.extend((data.len() as u32).to_le_bytes());
    out.extend(32u32.to_le_bytes()); // header size
    out.extend([0xff, 0xff]);
    out.extend(kind.to_le_bytes());
    out.extend([0xff, 0xff]);
    out.extend(name.to_le_bytes());
    out.extend(0u32.to_le_bytes()); // data version
    out.extend(flags.to_le_bytes());
    out.extend(LANGUAGE.to_le_bytes());
    out.extend(0u32.to_le_bytes()); // version
    out.extend(0u32.to_le_bytes()); // characteristics
    out.extend(data);
    out.resize(out.len().next_multiple_of(4), 0);
}

/// A compiled resource file holding the images as icons 1 to n and the group that lists them as icon group 1.
pub fn res(images: &[(usize, Vec<u8>)]) -> Vec<u8> {
    let mut out = Vec::new();
    entry(&mut out, 0, 0, 0, &[]); // the empty first entry that marks a 32-bit resource file
    let mut group = Vec::new();
    group.extend(0u16.to_le_bytes());
    group.extend(1u16.to_le_bytes()); // icons
    group.extend((images.len() as u16).to_le_bytes());
    for (index, (size, bytes)) in images.iter().enumerate() {
        let id = index as u16 + 1;
        entry(&mut out, RT_ICON, id, 0x1010, bytes);
        let side = if *size >= 256 { 0 } else { *size as u8 };
        group.extend([side, side, 0, 0]);
        group.extend(1u16.to_le_bytes());
        group.extend(32u16.to_le_bytes());
        group.extend((bytes.len() as u32).to_le_bytes());
        group.extend(id.to_le_bytes());
    }
    entry(&mut out, RT_GROUP_ICON, 1, 0x1030, &group);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u32_at(bytes: &[u8], at: usize) -> u32 {
        u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
    }

    fn u16_at(bytes: &[u8], at: usize) -> u16 {
        u16::from_le_bytes(bytes[at..at + 2].try_into().unwrap())
    }

    #[test]
    fn bitmaps_are_bottom_up_bgra_with_a_mask() {
        let mut rgba = vec![0u8; 2 * 2 * 4];
        rgba[..4].copy_from_slice(&[10, 20, 30, 40]); // top-left
        let dib = dib(2, &rgba);
        assert_eq!(u32_at(&dib, 0), 40);
        assert_eq!(u32_at(&dib, 8), 4, "the height counts the mask");
        assert_eq!(dib.len(), 40 + 16 + 4 * 2, "pixels, then one 4-byte mask row per row");
        assert_eq!(&dib[40 + 8..40 + 12], &[30, 20, 10, 40], "the top row is stored last, blue first");
    }

    #[test]
    fn the_resource_file_lists_every_image() {
        let images = images(|size| vec![255; size * size * 4], |_, _| b"\x89PNG fake".to_vec());
        assert_eq!(images.len(), SIZES.len() + 1);
        let res = res(&images);
        assert_eq!(&res[..8], &[0, 0, 0, 0, 32, 0, 0, 0], "the empty first entry");
        // Walk the entries: each is a 32-byte header and its data, padded to four bytes.
        let (mut at, mut kinds) = (0, Vec::new());
        while at < res.len() {
            let size = u32_at(&res, at) as usize;
            kinds.push((u16_at(&res, at + 10), u16_at(&res, at + 14)));
            at += (32 + size).next_multiple_of(4);
        }
        assert_eq!(at, res.len());
        assert_eq!(kinds.len(), images.len() + 2);
        assert_eq!(kinds.last(), Some(&(RT_GROUP_ICON, 1)));
        assert!(kinds[1..=images.len()].iter().enumerate().all(|(i, k)| *k == (RT_ICON, i as u16 + 1)));
        let group_size = u32_at(&res, res.len() - (6 + 14 * images.len()).next_multiple_of(4) - 32) as usize;
        assert_eq!(group_size, 6 + 14 * images.len());
    }
}
