#ifndef CLASSES_H
#define CLASSES_H

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
#include <algorithm>
#include <chrono>

// these setting must correspond to what is defined in util/tokens.py
#define BEGIN_TOKEN_STR         "BOS"
#define END_TOKEN_STR           "EOS"
#define SEP_TOKEN_STR           "SEP"
#define TRACK_EVENTS_CHAR       'R'
#define MEASURE_EVENTS_CHAR     'M'
#define POSITION_EVENTS_CHAR    'P'
#define TEMPO_EVENTS_CHAR       'T'
#define NOTE_EVENTS_CHAR        'N'
#define MULTI_NOTE_EVENT_CHAR   'U'

// for convenience
#define CONT_NOTE_EVENTS_STR    "N~"

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
    // onset:      Use 32 bits. If nth is 96, 0xffffffff of 96th notes is 728 minutes in speed of 240 beat per minute,
    //             which is enough for almost all music.
    uint32_t onset;
    static const uint32_t onsetLimit = 0xffffffff;
    // shapeIndex: Use 16 bits. The index of shape in the shapeDict. 0: DEFAULT_SHAPE_END, 1: DEFAULT_SHAPE_CONT
    //             This mean bpeIter cannot be greater than 0xffff - 2 = 65534
    uint16_t shapeIndex;
    static const uint16_t shapeIndexLimit = 0xffff - 2;
    uint8_t pitch;
    uint8_t unit; // time unit of shape
    uint8_t vel;

    // neighbor store relative index from this multinote to others
    // if neighbor > 0, any multinote in index (i+1) ~ (i+neighbor) where i is the index of current multinote
    // is this multinote's neighbor
    // because we only search toward greater index, it should only be positive integer or zero
    uint8_t neighbor;
    static const uint8_t neighborLimit = 0x7f;

    MultiNote(bool isCont, uint32_t o, uint8_t p, uint8_t d, uint8_t v);

    unsigned int getShapeIndex() const;

    void setShapeIndex(unsigned int s);

    unsigned int getOnset() const;

    void setOnset(unsigned int o);

    bool operator < (const MultiNote& rhs) const;
};

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
    size_t getMultiNoteCount();
    void sortAllTracks();
};

std::map<std::string, std::string> readParasFile(std::ifstream& paraFile);

Corpus readCorpusFile(std::ifstream& corpusFile, int nth);

void writeShapeVocabFile(std::ostream& vocabOutfile, const std::vector<Shape>& shapeDict);

void writeOutputCorpusFile(
    std::ostream& tokenizedCorpusFile,
    const Corpus& corpus,
    const std::vector<Shape>& shapeDict,
    int maxTrackNum
);

#endif
