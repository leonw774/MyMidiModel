#include "multinotes.hpp"
#include <fstream>
#include <string>
#include <cstring>
#include <sstream>
#include <vector>
#include <map>
#include <set>

typedef std::vector<RelNote> Shape;
typedef std::vector<MultiNote> Track;

struct TimeStructToken {

    uint16_t onset;

    // MSB ---------> LSB
    // T DDD NNNNNNNNNNNN
    // When T is 1, this is a tempo token, the tempo value is N. D is 0
    // When T is 0, this is a measure token, the denominator is 2 to the power of D and the numerator is N
    uint16_t data;

    TimeStructToken(uint16_t o, bool t, uint16_t n, uint16_t d);
    inline bool getT() const;
    inline int getD() const;
    inline int getN() const;
};

struct Corpus {
    // TimeStructures
    std::vector<std::vector<TimeStructToken>> piecesTS;
    // Multi-notes
    std::vector<std::vector<Track>> piecesMN;
    // Track numbers
    std::vector<std::vector<uint8_t>> piecesTN;

    void pushNewPiece();
    void shrink();
};

long b36strtol(const char* s);

std::string ltob36str(long x);

std::string shape2String(const Shape& s);

std::map<std::string, std::string> readParasFile(std::ifstream& paraFile);

Corpus readCorpusFile(std::ifstream& corpusFile, int nth, std::string positionMethod);

void writeTokenizedCorpusFile(std::ostream& tokenizedCorpusFile, Corpus corpus, std::string positionMethod);