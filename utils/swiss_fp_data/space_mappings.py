space_mapping = {
    "BALCONY": "TERRACE",
    "BATHROOM": "TOILET",
    "BEDROOM": "BEDROOM",
    "CORRIDOR": "CORRIDOR",
    "DINING": "DINING",
    "ELEVATOR": "SERVICE",
    "GARDEN": "GARDEN",
    "KITCHEN": "KITCHEN",
    "KITCHEN_DINING": "KITCHEN_DINING",
    "LIGHTWELL": "UTILITY",
    "LIVING_DINING": "LIVING_DINING",
    "LIVING_ROOM": "LIVING",
    "LOGGIA": "TERRACE",
    "OUTDOOR_VOID": "UTILITY",
    "PATIO": "TERRACE",
    "ROOM": "BEDROOM",
    "SHAFT": "UTILITY",
    "STAIRCASE": "SERVICE",
    "STOREROOM": "UTILITY",
    "STUDIO": "LIVING",
    "TECHNICAL_AREA": "UTILITY",
    "TERRACE": "TERRACE",
    "VOID": "UTILITY",
    "WINTERGARTEN": "TERRACE",
}

dw_mapping = {
    "DOOR": "DOOR",
    "ENTRANCE_DOOR": "DOOR",
    "WINDOW": "WINDOW",
}


def mapping(entity_type, entity_subtype):
    if entity_type == "area":
        return "SP-" + space_mapping[entity_subtype]
    if entity_type == "opening":
        return "DW-" + dw_mapping[entity_subtype]
    if entity_type == "separator" and entity_subtype == "WALL":
        return "PERIMETER"
    if entity_type == "feature":
        return "unknown"
    return "unknown"
