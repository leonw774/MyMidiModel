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
#include <unordered_set>
#include <algorithm>
#include <chrono>

// these setting must correspond to what is defined in tokens.py
#define BEGIN_TOKEN_STR         "BOS"
#define END_TOKEN_STR           "EOS"
#define SEP_TOKEN_STR           "SEP"
#define TRACK_EVENTS_CHAR       'R'
#define MEASURE_EVENTS_CHAR     'M'
#define POSITION_EVENTS_CHAR    'P'
#define TEMPO_EVENTS_CHAR       'T'
#define NOTE_EVENTS_CHAR        'N'
#define MULTI_NOTE_EVENTS_CHAR  'U'

// for convenience
#define CONT_NOTE_EVENTS_STR    "N~"

struct RelNote {
    uint8_t isCont : 1; // Lowest byte: lower 1 bit
    uint8_t relDur : 7; // Lowest byte: upper 7 bits
    int8_t relPitch;    // Middle byte
    uint8_t relOnset;   // Highest byte
    // Members are ordered such that it's value is:
    //      (MSB) aRandomByte relOnset relPitch relDur isCont (LSB)
    // when viewed as unsigned 32bit int

    static const uint8_t onsetLimit = 0x7f;
    static const uint8_t durLimit = 0x7f;

    RelNote();

    RelNote(uint8_t o, uint8_t p, uint8_t d, uint8_t c);

    inline void setRelDur(const uint8_t o);

    bool operator<(const RelNote& rhs) const;

    bool operator==(const RelNote& rhs) const;
};

typedef std::vector<RelNote> Shape;

int b36strtoi(const char* s);

std::string itob36str(int x);

std::string shape2str(const Shape& s);

unsigned int getMaxRelOffset(const Shape& s);

std::vector<Shape> getDefaultShapeDict();

template <>
struct std::hash<Shape> {
    size_t operator()(const Shape& s) const;
};

// typedef std::map<Shape, unsigned int> shape_counter_t;
typedef std::unordered_map<Shape, unsigned int> shape_counter_t;

struct MultiNote {
    uint32_t onset;
    // onsetLimit: When tpq is 12, 0x7ffffff ticks is 1,491,308 minutes in
    //             the tempo of 120 quarter note per minute,
    static const uint32_t onsetLimit = 0x7fffffff;
    // shapeIndex: The index of shape in the shapeDict.
    //             0: DEFAULT_SHAPE_REGULAR, 1: DEFAULT_SHAPE_CONT
    //             This mean iterNum cannot be greater than 0xffff - 2 = 65534
    uint16_t shapeIndex;
    static const uint16_t shapeIndexLimit = 0xffff - 2;
    uint8_t pitch;  // pitch shiht
    uint8_t stretch;// time stretch
    uint8_t vel;

    // This `neighbor` store relative index from this multinote to others.
    // If neighbor > 0, any multinote in index (i+1) ~ (i+neighbor) 
    // is the i-th multinote's neighbor.
    uint8_t neighbor;
    static const uint8_t neighborLimit = 0x7f;

    MultiNote(bool isCont, uint32_t o, uint8_t p, uint8_t d, uint8_t v);

    bool operator<(const MultiNote& rhs) const;

    bool operator==(const MultiNote& rhs) const;
};

typedef std::vector<MultiNote> Track;

void printTrack(
    const Track& track,
    const std::vector<Shape>& shapeDict,
    const size_t begin,
    const size_t length
);


struct TimeStructToken {
    uint32_t onset;

    // MSB ---------> LSB
    // T DDD NNNNNNNNNNNN
    // When T is 1, this is a tempo token, the tempo value is N. D is 0
    // When T is 0, this is a measure token,
    // the denominator is 2 to the power of D and the numerator is N
    uint16_t data;

    TimeStructToken(uint32_t o, bool t, uint16_t n, uint16_t d);
    bool isTempo() const;
    int getD() const;
    int getN() const;
};

struct Corpus {
    // Multi-notes
    std::vector<std::vector<Track>> mns;
    // Time structures
    std::vector<std::vector<TimeStructToken>> timeStructLists;
    // Track-program mappings
    std::vector<std::vector<uint8_t>> trackInstrMaps;

    void pushNewPiece();
    void shrink();
    size_t getMultiNoteCount();
    void sortAllTracks();
};

std::map<std::string, std::string> readParasFile(std::ifstream& paraFile);

Corpus readCorpusFile(std::ifstream& corpusFile, int tpq);

void writeShapeVocabFile(
    std::ostream& vocabOutfile,
    const std::vector<Shape>& shapeDict
);

void writeOutputCorpusFile(
    std::ostream& tokenizedCorpusFile,
    const Corpus& corpus,
    const std::vector<Shape>& shapeDict,
    int maxTrackNum
);

#endif
