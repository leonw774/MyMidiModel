#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <sstream>
#include <algorithm>
#include <vector>
#include <list>
#include <set>
#include <map>

#define DEBUG false
// #define DEBUG true
// #define DEBUG p == 0 && t == 0

using namespace std;

/*
def tokenint2str(x: int) -> str:
    if 0 <= x < TOKEN_INT2STR_BASE:
        return '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'[x]
    isneg = x < 0
    if isneg:
        x = -x
    b = ''
    while x:
        x, d = divmod(x, TOKEN_INT2STR_BASE)
        b = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'[d] + b
    return ('-' + b) if isneg else b
*/

string ltob36str(long x) {
    if (0 <= x && x < 36) return string(1, "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"[x]);
    bool isNeg = x < 0;
    if (isNeg) {
        x = -x;
    }
    string r;
    while (x) {
        r = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"[x%36] + r;
        x /= 36;
    }
    return (isNeg ? "-" : "") + r;
}

int gcd (int a, int b) {
    int tmp;
    if (a == b) return a;
    if (a == 1 || b == 1) return 1;
    while (b > 0) {
        tmp = b;
        b = a % b;
        a = tmp;
    }
    return a;
}

int gcd (int* arr, unsigned int size) {
    int g = abs(arr[0]);
    for (int i = 1; i < size; ++i) {
        g = gcd(g, abs(arr[i]));
        if (g == 1) return 1;
    }
    return g;
}

struct Note {
    // pitch, duration, velocity, track number
    uint32_t onset;
    int8_t pitch;
    uint8_t dur;
    uint8_t vel;
    uint8_t trn;
};

struct RelNote {
    uint8_t isContAndRelOnset;
    int8_t relPitch;
    uint8_t relDur;

    RelNote() : isContAndRelOnset(0), relPitch(0), relDur(0) {};

    RelNote(uint8_t c, uint8_t o, uint8_t p, uint8_t d) {
        isContAndRelOnset = (c ? 0x80 : 0x00) | (o & 0x7f);
        relPitch = p;
        relDur = d;
    }

    inline bool isCont() const {
        return isContAndRelOnset >> 7;
    }

    inline unsigned int getRelOnset() const {
        return isContAndRelOnset & 0x7f;
    }

    inline void setRelOnset(const uint8_t o) {
        isContAndRelOnset = (isContAndRelOnset & 0x80) | (o & 0x7f);
    }

    inline unsigned int getRelPitch() const {
        return relPitch;
    }

    inline unsigned int getRelDur() const {
        return relDur;
    }

    inline bool operator < (const RelNote& right) const {
        if (getRelOnset() + relDur == right.getRelOnset() + right.relDur) {
            if (getRelOnset() == right.getRelOnset()) {
                return relPitch < right.relPitch;
            }
            return getRelOnset() < right.getRelOnset();
        }
        return getRelOnset() + relDur < right.getRelOnset() + right.relDur;
    } 
};

typedef vector<RelNote> Shape;

Shape DEFAULT_SHAPE_END({RelNote(0, 0, 0, 1)});
Shape DEFAULT_SHAPE_CONT({RelNote(1, 0, 0, 1)});

struct ShapeOccurence {
    // piece_idx use upper 24 bit, track_idx use lower 8 bits
    uint32_t pieceTrackIdx;
    uint16_t mnLeftIdx;
    uint16_t mnRightIdx;

    ShapeOccurence (unsigned int p, unsigned int t, unsigned int l, unsigned int r) {
        pieceTrackIdx = (p << 8) | (t & 0xff);
        mnLeftIdx = l;
        mnRightIdx = r;
    }

    inline unsigned int getPidx() const {
        return pieceTrackIdx >> 8;
    }

    inline unsigned int getTidx() const {
        return pieceTrackIdx & 0xff;
    }

    inline unsigned int getLidx() const {
        return mnLeftIdx;
    }

    inline unsigned int getRidx() const {
        return mnRightIdx;
    }

    inline bool operator < (const ShapeOccurence& right) const {
        if (pieceTrackIdx == right.pieceTrackIdx) {
            return mnLeftIdx < right.mnLeftIdx;
        }
        return pieceTrackIdx < right.pieceTrackIdx;
    }

    inline bool operator == (const ShapeOccurence& right) const {
        return pieceTrackIdx == right.pieceTrackIdx && mnLeftIdx == right.mnLeftIdx;
    }
};

struct MultiNote {
    Shape* shape; // pointed to a shape in dictionary or DEFAULT_SHAPE
    uint32_t onset;
    uint8_t pitch;
    uint8_t unit; // time unit of shape
    uint8_t vel;

    // neighbors store relative index from this multinote to others
    // because we only search toward greater index, it should be positive integer
    vector<unsigned int> neighbors;

    MultiNote(Note n) {
        onset = n.onset;
        if (n.pitch < 0) {
            pitch = -n.pitch;
            shape = &DEFAULT_SHAPE_CONT; // (1, 0, 0, 1)
        }
        else {
            pitch = n.pitch;
            shape = &DEFAULT_SHAPE_END; // (0, 0, 0, 1)
        }
        unit = n.dur;
        vel = n.vel;
        neighbors.clear();
    }

    bool operator < (const MultiNote& right) {
        if (onset == right.onset) {
            return pitch < right.pitch;
        }
        return onset < right.onset;
    }
};


typedef vector<vector<MultiNote>> Piece;

string shape2String(const Shape& s) {
    stringstream r;
    for (int j = 0; j < s.size(); ++j) {
        r <<        ltob36str(s[j].getRelOnset())
          << "," << ltob36str(s[j].relPitch)
          << "," << ltob36str(s[j].relDur) << (s[j].isCont() ? "~" : "") << ";";
    }
    return r.str();
}

void printTrack(const vector<MultiNote>& track, const size_t begin, const size_t length) {
    for (int i = begin; i < begin + length; ++i) {
        cout << i << " - Shape=" << shape2String(*(track[i].shape));
        cout << " onset=" << (int) track[i].onset
                << " basePitch=" << (int) track[i].pitch
                << " timeUnit=" << (int) track[i].unit
                << " velocity=" << (int) track[i].vel;
        cout << " neighbors(" << track[i].neighbors.size() << "):";
        for (auto n : track[i].neighbors) {
            cout << n << ','; 
        }
        cout << endl;
    }
}

void updateNeighbors(vector<Piece>& corpus, long maxDurationTick) {
    // for each piece
    for (int i = 0; i < corpus.size(); ++i) {
        // for each track
        for (int j = 0; j < corpus[i].size(); ++j) {
            // for each multinote
            for (int k = 0; k < corpus[i][j].size(); ++k) {
                // printTrack(corpus[i][j], k, 1);
                uint32_t onsetTime = corpus[i][j][k].onset;
                RelNote latestRelNote = (*corpus[i][j][k].shape).back();
                uint32_t relOffsetTime = latestRelNote.getRelOnset() + latestRelNote.relDur;
                uint32_t offsetTime = corpus[i][j][k].onset + relOffsetTime * corpus[i][j][k].unit;
                uint32_t immdAfterOnset = -1;
                int n = 1;
                corpus[i][j][k].neighbors.clear();
                while (k+n < corpus[i][j].size()) {
                    // has possibility to make timeUnit > maxDurationTick
                    if ((corpus[i][j][k+n]).onset > onsetTime + maxDurationTick) {
                        break;
                    }
                    // simultaneous
                    if ((corpus[i][j][k+n]).onset == onsetTime) {
                        if (corpus[i][j][k].pitch > (corpus[i][j][k+n]).pitch) {
                            cout << "Pitches are not sorted ascending at:"
                                 << i << ',' << j << endl;
                            printTrack(corpus[i][j], k, k+n);
                            exit(1);
                        }
                        if (corpus[i][j][k].vel == (corpus[i][j][k+n]).vel) {
                            corpus[i][j][k].neighbors.push_back(n);
                        }
                    }
                    // overlapping
                    else if ((corpus[i][j][k+n]).onset < offsetTime) {
                        if (corpus[i][j][k].vel == (corpus[i][j][k+n]).vel) {
                            corpus[i][j][k].neighbors.push_back(n);
                        }
                    }
                    // immediately after
                    else if (immdAfterOnset == -1 || (corpus[i][j][k+n]).onset == immdAfterOnset) {
                        immdAfterOnset = (corpus[i][j][k+n]).onset;
                        if (corpus[i][j][k].vel == (corpus[i][j][k+n]).vel) {
                            corpus[i][j][k].neighbors.push_back(n);
                        }
                    }
                    else {
                        break;
                    }
                    n++;
                }
            }
        }
    }
}

Shape getShapeOfMultiNotePair(const MultiNote& mnleft, const MultiNote& mnright) {
    int pairSize = mnleft.shape->size() + mnright.shape->size();
    Shape pairShape;
    vector<int> onsetTimes, durTimes;
    pairShape.resize(pairSize);
    onsetTimes.reserve(pairSize);
    durTimes.reserve(pairSize);
    for (int i = 0; i < mnleft.shape->size(); ++i) {
        onsetTimes.push_back((*mnleft.shape)[i].getRelOnset() * mnleft.unit);
        durTimes.push_back((*mnleft.shape)[i].relDur * mnleft.unit);
    }
    for (int i = 0; i < mnright.shape->size(); ++i) {
        onsetTimes.push_back((*mnright.shape)[i].getRelOnset() * mnright.unit + mnright.onset - mnleft.onset);
        durTimes.push_back((*mnright.shape)[i].relDur * mnright.unit);
    }
    int newUnit = gcd(
        gcd(onsetTimes.data(), pairSize),
        gcd(durTimes.data(), pairSize)
    );
    // cout << "newUnit:" << newUnit << endl;
    for (int i = 0; i < pairSize; ++i) {
        pairShape[i].setRelOnset(onsetTimes[i] / newUnit);
        pairShape[i].relDur = durTimes[i] / newUnit;
        if (i < mnleft.shape->size()) {
            pairShape[i].relPitch = (*mnleft.shape)[i].relPitch;
        }
        else {
            int j = i - mnleft.shape->size();
            pairShape[i].relPitch = (*mnright.shape)[j].relPitch + mnright.pitch - mnleft.pitch; 
        }
    }
    sort(pairShape.begin(), pairShape.end());
    return pairShape;
}


int main(int argc, char *argv[]) {
    // validate args
    if (argc != 4) {
        cout << "Must have 3 arguments: [maxDurationTick] [bpeIter] [inputFilePath]" << endl;
        return 1;
    }
    char* pEnd;

    long maxDurationTick = strtol(argv[1], &pEnd, 10);
    if (argv[1]) {
        cout << "maxDurationTick: " << maxDurationTick << endl;
    }
    else {
        cout << "First arguments [maxDurationTick] is not convertable by strtol in base 10: " << endl;
        return 1;
    }

    long bpeIter = strtol(argv[2], &pEnd, 10);
    if (argv[2]) {
        cout << "bpeIter: " << bpeIter << endl;
    }
    else {
        cout << "Sencond arguments [bpeIter] is not convertable by strtol in base 10: " << endl;
        return 1;
    }

    ifstream infile(argv[3], ios::in | ios::binary);
    if (!infile.is_open()) {
        cout << "Failed to open input file: " << argv[3] << endl;
        return 1;
    }
    cout << "Input file: " << argv[3] << endl;

    string vocabOutfileName(argv[3]);
    vocabOutfileName.append("_vocab");
    ofstream vocabOutfile(vocabOutfileName, ios::out | ios::trunc);
    if (!vocabOutfile.is_open()) {
        cout << "Failed to open vocab output file: " << vocabOutfileName << endl;
        return 1;
    }

    string textOutfileName(argv[3]);
    textOutfileName.append("_mntext");
    ofstream textOutfile(textOutfileName, ios::out | ios::trunc);
    if (!textOutfile.is_open()) {
        cout << "Failed to open text output file: " << textOutfileName << endl;
        return 1;
    }

    // read & construct data
    infile.seekg(0, ios::end);
    streampos filesize = infile.tellg();
    infile.seekg(0, ios::beg);
    cout << "File size: " << filesize << " bytes" << endl;
    if (filesize % 8 != 0 || filesize == 0) {
        cout << "Broken file size" << endl;
        return 1;
    } 

    Note* noteSeq = new Note[filesize/sizeof(Note)];
    infile.read((char*) noteSeq, filesize);

    // for (int i = 0; i < 8; ++i) {
    //     cout << (int) noteSeq[i].onset << ','
    //          << (int) noteSeq[i].pitch << ','
    //          << (int) noteSeq[i].dur << ','
    //          << (int) noteSeq[i].vel << ','
    //          << (int) noteSeq[i].trn << endl;
    // }

    // use list to store shape vocabs because we dont want the address of each shape changes
    list<Shape> shapeDict;
    vector<Piece> corpus;
    int beginPos = 0;
    uint8_t maxTrn = 0;
    for (int i = 0; i < filesize/sizeof(Note); ++i) {
        if (((uint64_t*)noteSeq)[i] == 0) {
            corpus.push_back(Piece(maxTrn+1, vector<MultiNote>()));
            for (int j = beginPos; j < i; ++j) {
                corpus.back()[noteSeq[j].trn].push_back(MultiNote(noteSeq[j]));
            }
            beginPos = i + 1;
            maxTrn = 0;
        }
        else {
            // count tracks
            maxTrn = max(maxTrn, noteSeq[i].trn);
        }
    }

    // sort and count notes
    size_t multinoteCount = 0;
    for (unsigned int i = 0; i < corpus.size(); ++i) {
        for (unsigned int j = 0; j < corpus[i].size(); ++j) {
            multinoteCount += corpus[i][j].size();
            sort(corpus[i][j].begin(), corpus[i][j].end());
        }
    }
    cout << "Start multinote count: " << multinoteCount << endl;

    // init neighbors
    updateNeighbors(corpus, maxDurationTick);
    // printTrack(corpus[0][0], 0, 8);

    map<Shape, set<ShapeOccurence>> shapeOccurs;
    for (int iterCount = 0; iterCount < bpeIter; ++iterCount) {
        cout << "iter:" << iterCount << ", ";
        // count shape frequency
        
        // for each piece
        for (unsigned int i = 0; i < corpus.size(); ++i) {
            // for each track
            for (unsigned int j = 0; j < corpus[i].size(); ++j) {
                // for each multinote
                for (unsigned int k = 0; k < corpus[i][j].size(); ++k) {
                    // for each neighbor
                    for (auto it = corpus[i][j][k].neighbors.cbegin(); it != corpus[i][j][k].neighbors.cend(); it++) {
                        // if (DEBUG) cout << i << "," << j << "," << k << "->" << k+(*it);
                        Shape s = getShapeOfMultiNotePair(corpus[i][j][k], corpus[i][j][k+(*it)]);
                        // if (DEBUG)  cout << " - " << shape2String(s) << endl;
                        if (shapeOccurs.count(s) == 0) {
                            shapeOccurs.insert({s, set<ShapeOccurence>()});
                        }
                        shapeOccurs[s].insert(ShapeOccurence(i, j, k, k+(*it)));
                    }
                }
            }
        }
        cout << "Find " << shapeOccurs.size() << " unique pairs" << ", ";
    
        // add shape with highest frequency into shapeDict
        Shape maxFreqShape;
        unsigned int maxFreq = 0;
        for (auto it = shapeOccurs.cbegin(); it != shapeOccurs.cend(); it++) {
            if (maxFreq < (*it).second.size()) {
                maxFreqShape = (*it).first;
                maxFreq = (*it).second.size();
            }
            // cout << shape2String((*it).first) << endl;
        }
        shapeDict.push_back(maxFreqShape);
        cout << "Add new shape: " << shape2String(maxFreqShape) << " with freq=" << maxFreq << endl;

        // merge MultiNotes with new added shape
        Shape* pairShapePtr = &(shapeDict.back());
        for (auto it = shapeOccurs[maxFreqShape].cbegin(); it != shapeOccurs[maxFreqShape].cend(); it++) {
            unsigned int p = (*it).getPidx();
            unsigned int t = (*it).getTidx();
            unsigned int l = (*it).getLidx();
            unsigned int r = (*it).getRidx();

            if (corpus[p][t][l].neighbors.size() == 0) {
                throw runtime_error("Left multinote does not have neighbor");
            }

            if (DEBUG) {
                cout << "occur: " << p << ',' << t << ':' << l << '-' << r << endl;
                printTrack(corpus[p][t], l, 1);
                printTrack(corpus[p][t], r, 1);
            }

            // change left multinote to merged multinote
            corpus[p][t][l].unit = (*(corpus[p][t][l].shape))[0].relDur * corpus[p][t][l].unit / (*pairShapePtr)[0].relDur;
            corpus[p][t][l].shape = pairShapePtr;

            if (DEBUG) {
                cout << "NEW: "; printTrack(corpus[p][t], l, 1);
                cout << "--------\n";
            }
            // dont need to delete shape in multinote because all those shapes are either pointed to
            // global variable DEFAULT_SHAPE_* or a shape object in shapeDict
        }
        // remove right multinote after all left node are changed into merged multinote
        // we erase element in reverse order to make sure index was not changed when erasing
        for (auto it = shapeOccurs[maxFreqShape].rbegin(); it != shapeOccurs[maxFreqShape].rend(); it++) {
            unsigned int p = (*it).getPidx();
            unsigned int t = (*it).getTidx();
            unsigned int r = (*it).getRidx();
            // https://stackoverflow.com/questions/3487717/erasing-multiple-objects-from-a-stdvector
            corpus[p][t][r] = corpus[p][t].back();
            corpus[p][t].pop_back();
        }

        // sort effected tracks, which could be all, but nah
        set<uint32_t> sortedPieceTrackId;
        for (auto it = shapeOccurs[maxFreqShape].begin(); it != shapeOccurs[maxFreqShape].end(); it++) {
            uint32_t pt = (*it).pieceTrackIdx;
            if (!sortedPieceTrackId.count(pt)) {
                unsigned int p = (*it).getPidx();
                unsigned int t = (*it).getTidx();
                sortedPieceTrackId.insert(pt);
                sort(corpus[p][t].begin(), corpus[p][t].end());
            }
        }

        shapeOccurs.clear();
        updateNeighbors(corpus, maxDurationTick);
    }

    multinoteCount = 0;
    for (int i = 0; i < corpus.size(); ++i) {
        for (int j = 0; j < corpus[i].size(); ++j) {
            multinoteCount += corpus[i][j].size();
        }
    }
    cout << "Final multinote count: " << multinoteCount << endl;

    cout << "Final non-default shape cout: " << shapeDict.size() << endl;

    // write vocab file
    stringstream shapeDictStr;
    for (auto it = shapeDict.cbegin(); it != shapeDict.cend(); it++) {
        shapeDictStr << 'S' << shape2String(*it) << endl;
    }
    vocabOutfile << shapeDictStr.str();
    cout << "Write vocab file done." << endl;

    // write notes with onset information into text file
    // Format: onset_time_in_tick '@' (S-format multinote | N-format multinote)
    // for each piece
    for (int i = 0; i < corpus.size(); ++i) {
        // keep track of each track's next notes's index
        vector<unsigned int> curIndex(corpus[i].size(), 0);
        while (1) {
            unsigned int minOnset = -1;
            unsigned int minOnsetTrack = -1;
            for (int j = 0; j < corpus[i].size(); ++j) {
                if (curIndex[j] == -1) continue;
                if (minOnset > corpus[i][j][curIndex[j]].onset) {
                    minOnset = corpus[i][j][curIndex[j]].onset;
                    minOnsetTrack = j;
                }
            }
            string shapeStr = shape2String(*(corpus[i][minOnsetTrack][curIndex[minOnsetTrack]].shape));
            if (shapeStr == "0,0,1;") {
                shapeStr = "N";
            }
            else if (shapeStr == "0,0,1~;") {
                shapeStr = "N~";
            }
            else {
                shapeStr = 'S' + shapeStr;
            }
            stringstream s;
            s << ltob36str(corpus[i][minOnsetTrack][curIndex[minOnsetTrack]].onset) << '@'
              << shapeStr
              << ':' << ltob36str(corpus[i][minOnsetTrack][curIndex[minOnsetTrack]].pitch)
              << ':' << ltob36str(corpus[i][minOnsetTrack][curIndex[minOnsetTrack]].unit)
              << ':' << ltob36str(corpus[i][minOnsetTrack][curIndex[minOnsetTrack]].vel)
              << ':' << ltob36str(minOnsetTrack) << ' ';
            textOutfile << s.str();

            // update index
            curIndex[minOnsetTrack] += 1;
            if (curIndex[minOnsetTrack] >= corpus[i][minOnsetTrack].size()) {
                curIndex[minOnsetTrack] = -1;
            }
            
            int sumOfIndices = 0;
            for (int j = 0; j < curIndex.size(); ++j)
                sumOfIndices += curIndex[j];
            if (sumOfIndices == -1 * curIndex.size())
                break;
        }
        textOutfile << endl;
    }
    cout << "Write multinote text file done." << endl;

    return 0;
}