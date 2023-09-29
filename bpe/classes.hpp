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
    uint8_t relOnset;
    int8_t relPitch;
    // upper 7 bits used by relDur and lower 1 bit used by isCont
    // It means the maxDuration is 127 at maximum
    uint8_t relDurAndIsCont;
    static const uint8_t onsetLimit = 0xff;
    static const uint8_t durLimit = 0x7f;

    RelNote();

    RelNote(uint8_t c, uint8_t o, uint8_t p, uint8_t d);

    // inline bool isCont() const;

    // inline unsigned int getRelOnset() const;

    // inline void setRelOnset(const uint8_t o);

    bool operator<(const RelNote& rhs) const;

    bool operator==(const RelNote& rhs) const;
};

// using macro is faster
#define GET_IS_CONT(r)      (r.relDurAndIsCont & 1)
#define GET_REL_DUR(r)      (r.relDurAndIsCont >> 1)
#define SET_REL_DUR(r, o)   (r.relDurAndIsCont = (r.relDurAndIsCont & 1) | (o << 1))

typedef std::vector<RelNote> Shape;

int b36strtoi(const char* s);

std::string itob36str(int x);

std::string shape2str(const Shape& s);

unsigned int getMaxRelOffset(const Shape& s);

std::vector<Shape> getDefaultShapeDict();

typedef std::map<Shape, unsigned int> shape_counter_t;

struct MultiNote {
    // onset:       Use 32 bits. When nth is 32, 0xffffffff of 32th notes is 4,473,924 minutes in
    //              the tempo of 120 quarter note per minute,
    uint32_t onset;
    static const uint32_t onsetLimit = 0xffffffff;
    // shapeIndex:  Use 16 bits. The index of shape in the shapeDict.
    //              0: DEFAULT_SHAPE_REGULAR, 1: DEFAULT_SHAPE_CONT
    //              This mean iterNum cannot be greater than 0xffff - 2 = 65534
    uint16_t shapeIndex;
    static const uint16_t shapeIndexLimit = 0xffff - 2;
    uint8_t pitch;  // pitch shiht
    uint8_t stretch;// time stretch
    uint8_t vel;

    // `neighbor` store relative index from this multinote to others
    // if neighbor > 0, any multinote in index (i+1) ~ (i+neighbor) where i is the index of current multinote
    // is this multinote's neighbor
    // because we only search toward greater index, it should only be positive integer or zero
    uint8_t neighbor;
    static const uint8_t neighborLimit = 0x7f;

    MultiNote(bool isCont, uint32_t o, uint8_t p, uint8_t d, uint8_t v);

    bool operator<(const MultiNote& rhs) const;

    bool operator==(const MultiNote& rhs) const;
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
    std::vector<std::vector<TimeStructToken>> timeStructs;
    // Multi-notes
    std::vector<std::vector<Track>> mns;
    // Track-program mapping
    std::vector<std::vector<uint8_t>> trackInstrMap;

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
