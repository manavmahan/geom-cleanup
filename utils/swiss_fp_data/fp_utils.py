from .helper import SpaceName, DoorWindow, Boundary


def hex_to_rgb(hexcode):
    """Convert hexcode to RGB tuple."""
    hexcode = hexcode.lstrip("#")
    r = int(hexcode[0:2], 16)
    g = int(hexcode[2:4], 16)
    b = int(hexcode[4:6], 16)
    return (r, g, b)


class Mapping:
    """Index to name mapping."""

    mapping = (("PERIMETER", 0),)  # this will turn into walls
    NameToIndex = dict((k, v) for k, v in mapping)
    IndexToName = dict((v, k) for k, v in mapping)
    IndexToColor = dict((v, "#000000") for k, v in mapping)

    start_names = len(mapping)
    for cls in SpaceName.__members__:
        mapping += ((SpaceName[cls], len(mapping)),)
    for cls in DoorWindow.__members__:
        mapping += ((DoorWindow[cls], len(mapping)),)
    for cls in Boundary.__members__:
        mapping += ((Boundary[cls], len(mapping)),)

    NameToIndex.update(dict((k.name, v) for k, v in mapping[start_names:]))
    IndexToName.update(dict((v, k.name) for k, v in mapping[start_names:]))
    IndexToColor.update(dict((v, k.value["color"]) for k, v in mapping[start_names:]))


class SpaceMapping:
    """Space mapping."""

    mapping = ()
    for cls in SpaceName.__members__:
        mapping += ((SpaceName[cls], len(mapping)),)
    NameToIndex = dict((k.name, v) for k, v in mapping)
    IndexToName = dict((v, k) for k, v in mapping)
    IndexToColor = dict((v, hex_to_rgb(k.value["color"])) for k, v in mapping)
