#include "multinotes.hpp"

RelNote::RelNote() : isContAndRelOnset(0), relPitch(0), relDur(0) {};

RelNote::RelNote(uint8_t c, uint8_t o, uint8_t p, uint8_t d) {
    isContAndRelOnset = (c ? 0x80 : 0x00) | (o & 0x7f);
    relPitch = p;
    relDur = d;
}

inline bool RelNote::isCont() const {
    return isContAndRelOnset >> 7;
}

inline void RelNote::setCont(bool c) {
    if (c) {
        isContAndRelOnset |= 0x80;
    }
    else {
        isContAndRelOnset &= 0x7f;
    }
}

inline unsigned int RelNote::getRelOnset() const {
    return isContAndRelOnset & 0x7f;
}

inline void RelNote::setRelOnset(const uint8_t o) {
    isContAndRelOnset = (isContAndRelOnset & 0x80) | (o & 0x7f);
}

inline bool RelNote::operator < (const RelNote& rhs) const {
    // sort on onset first, and then pitch, finally duration.
    if (getRelOnset() == rhs.getRelOnset()) {
        if (relPitch == rhs.relPitch) {
            return relDur < rhs.relDur;
        }
        return relPitch < rhs.relPitch;
    }
    return getRelOnset() < rhs.getRelOnset();
}

inline bool RelNote::operator == (const RelNote& rhs) const {
    return (isContAndRelOnset == rhs.isContAndRelOnset
        && relPitch == rhs.relPitch
        && relDur == rhs.relDur);
}


MultiNote::MultiNote(bool isCont, uint32_t o, uint8_t p, uint8_t d, uint8_t v) {
    if (isCont) {
        // shape index = 1 -> {RelNote(1, 0, 0, 1)}
        shapeIndexAndOnset = 0x100000u | (o & 0x0fffffu);
    }
    else {
        // shape index = 0 -> {RelNote(0, 0, 0, 1)}
        shapeIndexAndOnset = (o & 0x0fffffu);
    }
    pitch = p;
    unit = d;
    vel = v;
    neighbor = 0;
}

inline unsigned int MultiNote::getShapeIndex() const {
    return shapeIndexAndOnset >> 20;
}

inline void MultiNote::setShapeIndex(unsigned int s) {
    shapeIndexAndOnset = (shapeIndexAndOnset & 0x0fffffu) | (s << 20);
}

inline unsigned int MultiNote::getOnset() const {
    return shapeIndexAndOnset & 0x0fffffu;
}

inline void MultiNote::setOnset(unsigned int o) {
    shapeIndexAndOnset = ((shapeIndexAndOnset >> 20) << 20) | (o & 0x0fffffu);
}

inline bool MultiNote::operator < (const MultiNote& rhs) const {
    if ((shapeIndexAndOnset & 0x0fffffu) == rhs.getOnset()) {
        return pitch < rhs.pitch;
    }
    return (shapeIndexAndOnset & 0x0fffffu) < rhs.getOnset();
}
