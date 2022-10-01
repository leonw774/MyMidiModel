#ifndef CORPUS_H
#define CORPUS_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <queue>
#include <string>
#include <cstring>
#include <map>
#include <unordered_map>
#include <set>
#include <chrono>

struct RelNote {
    uint8_t isContAndRelOnset;
    int8_t relPitch;
    uint8_t relDur;
    static const uint8_t onsetLimit = 0x7f;

    RelNote();

    RelNote(uint8_t c, uint8_t o, uint8_t p, uint8_t d);

    bool isCont() const;

    void setCont(bool c);

    unsigned int getRelOnset() const;

    void setRelOnset(const uint8_t o);

    bool operator < (const RelNote& rhs) const;

    bool operator == (const RelNote& rhs) const;
};

// inline method should be implemented in header

inline bool RelNote::isCont() const {
    return isContAndRelOnset >> 7;
}

inline void RelNote::setCont(bool c) {
    isContAndRelOnset = (c ? (isContAndRelOnset | 0x80) : (isContAndRelOnset & 0x7f));
}

inline unsigned int RelNote::getRelOnset() const {
    return isContAndRelOnset & 0x7f;
}

inline void RelNote::setRelOnset(const uint8_t o) {
    isContAndRelOnset = (isContAndRelOnset & 0x80) | (o & 0x7f);
}

typedef std::vector<RelNote> Shape;

int b36strtoi(const char* s);

std::string itob36str(int x);

std::string shape2str(const Shape& s);

unsigned int findMaxRelOffset(const Shape& s);

std::vector<Shape> getDefaultShapeDict();


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

    unsigned int getShapeIndex() const;

    void setShapeIndex(unsigned int s);

    unsigned int getOnset() const;

    void setOnset(unsigned int o);

    bool operator < (const MultiNote& rhs) const;
};

// inline method should be implemented in header

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

typedef std::vector<MultiNote> Track;

void printTrack(const Track& track, const std::vector<Shape>& shapeDict, const size_t begin, const size_t length);


struct TimeStructToken {

    uint32_t onset;

    // MSB ---------> LSB
    // T DDD NNNNNNNNNNNN
    // When T is 1, this is a tempo token, the tempo value is N. D is 0
    // When T is 0, this is a measure token, the denominator is 2 to the power of D and the numerator is N
    uint16_t data;

    TimeStructToken(uint32_t o, bool t, uint16_t n, uint16_t d);
    bool getT() const;
    int getD() const;
    int getN() const;
};

struct Corpus {
    // Time structures
    std::vector<std::vector<TimeStructToken>> piecesTS;
    // Multi-notes
    std::vector<std::vector<Track>> piecesMN;
    // Track-program mapping
    std::vector<std::vector<uint8_t>> piecesTP;

    void pushNewPiece();
    void shrink();
};

std::map<std::string, std::string> readParasFile(std::ifstream& paraFile);

Corpus readCorpusFile(std::ifstream& corpusFile, int nth, std::string positionMethod);

void writeShapeVocabFile(std::ostream& vocabOutfile, const std::vector<Shape>& shapeDict);

void writeOutputCorpusFile(
    std::ostream& tokenizedCorpusFile,
    const Corpus& corpus,
    const std::vector<Shape>& shapeDict,
    int maxTrackNum,
    const std::string& positionMethod
);

#endif
