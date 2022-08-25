#include <iostream>

struct RelNote {
    uint8_t isContAndRelOnset;
    int8_t relPitch;
    uint8_t relDur;

    RelNote() : isContAndRelOnset(0), relPitch(0), relDur(0) {};

    RelNote(uint8_t c, uint8_t o, uint8_t p, uint8_t d);

    inline bool isCont() const;

    inline void setCont(bool c);

    inline unsigned int getRelOnset() const;

    inline void setRelOnset(const uint8_t o);

    inline bool operator < (const RelNote& rhs) const;

    inline bool operator == (const RelNote& rhs) const;
};


struct MultiNote {
    // shapeIndex: High 12 bits. The index of shape in the shapeDict. 0: DEFAULT_SHAPE_END, 1: DEFAULT_SHAPE_CONT
    //             This mean bpeIter cannot be greater than 0xfff - 2 = 2045
    // onset:      Low 20 bits. If nth is 96, 0xfffff of 96th notes is 182 minutes in speed of 240 beat per minute,
    //             which is enough for almost all music.
    uint32_t shapeIndexAndOnset;
    uint8_t pitch;
    uint8_t unit; // time unit of shape
    uint8_t vel;

    // neighbor store relative index from this multinote to others
    // if neighbor > 0, any multinote in (i+1) ~ (i+neighbor) where i is the index of current multinote
    // is this multinote's neighbor
    // because we only search toward greater index, it should only be positive integer
    uint8_t neighbor;

    MultiNote(bool isCont, uint32_t o, uint8_t p, uint8_t d, uint8_t v);

    inline unsigned int getShapeIndex() const;

    inline void setShapeIndex(unsigned int s);

    inline unsigned int getOnset() const;

    inline void setOnset(unsigned int o);

    inline bool operator < (const MultiNote& rhs) const;
};
