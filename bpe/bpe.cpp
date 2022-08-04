#include <iostream>
#include <fstream>
#include <cstring>
#include <ctime>
#include <string>
#include <sstream>
#include <algorithm>
#include <vector>
#include <list>
#include <set>
#include <map>
#include "omp.h"

// #define DEBUG false
// #define DEBUG true
#define DEBUG i == 0 && j == 0

using namespace std;

long b36strtol(const char* s) {
    unsigned int l = strlen(s);
    bool isNeg = s[0] == '-';
    long r = 0;
    for (unsigned int i = (isNeg ? 1 : 0); i < l; ++i) {
        if ('0' <= s[i] && s[i] <= '9') {
            r = (r << 5) + (r << 2) + s[i] - '0'; // r * 36 --> r = (r * 32 + r * 4)
        }
        else if ('A' <= s[i] && s[i] <= 'Z') {
            r = (r << 5) + (r << 2) + s[i] - 'A' + 10;
        }
    }
    return (isNeg ? -r : r);
}

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
    if (a == b) return a;
    if (a == 1 || b == 1) return 1;
    int tmp;
    while (b > 0) {
        tmp = b;
        b = a % b;
        a = tmp;
    }
    return a;
}

int gcd (int* arr, unsigned int size) {
    int g = arr[0];
    for (int i = 1; i < size; ++i) {
        g = gcd(g, arr[i]);
        if (g == 1) return 1;
    }
    return g;
}


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

    inline void setCont(bool c) {
        if (c) {
            isContAndRelOnset |= 0x80;
        }
        else {
            isContAndRelOnset &= 0x7f;
        }
    }

    inline unsigned int getRelOnset() const {
        return isContAndRelOnset & 0x7f;
    }

    inline void setRelOnset(const uint8_t o) {
        isContAndRelOnset = (isContAndRelOnset & 0x80) | (o & 0x7f);
    }

    inline bool operator < (const RelNote& rhs) const {
        // sort on offset first, so that when we want to know the length of shape
        // we can just sort it and look at last relNote
        if ((isContAndRelOnset & 0x7f) + relDur == rhs.getRelOnset() + rhs.relDur) {
            if ((isContAndRelOnset & 0x7f) == rhs.getRelOnset()) {
                return relPitch < rhs.relPitch;
            }
            return (isContAndRelOnset & 0x7f) < rhs.getRelOnset();
        }
        return (isContAndRelOnset & 0x7f) + relDur < rhs.getRelOnset() + rhs.relDur;
    }

    inline bool operator == (const RelNote& rhs) const {
        return (isContAndRelOnset == rhs.isContAndRelOnset
            && relPitch == rhs.relPitch
            && relDur == rhs.relDur);
    }
};

typedef vector<RelNote> Shape;

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

    MultiNote(bool isCont, uint32_t o, uint8_t p, uint8_t d, uint8_t v) {
        shapeIndexAndOnset = (o & 0x0fffff);
        if (isCont) {
            // shape index = 1 -> {RelNote(1, 0, 0, 1)}
            shapeIndexAndOnset = 0x100000 | (o & 0x0fffff);
        }
        pitch = p;
        unit = d;
        vel = v;
        neighbor = 0;
    }

    inline unsigned int getShapeIndex() const {
        return shapeIndexAndOnset >> 20;
    }

    inline void setShapeIndex(unsigned int s) {
        shapeIndexAndOnset = (shapeIndexAndOnset & 0x0fffff) | (s << 20);
    }

    inline unsigned int getOnset() const {
        return shapeIndexAndOnset & 0x0fffff;
    }

    inline void setOnset(unsigned int o) {
        shapeIndexAndOnset = ((shapeIndexAndOnset >> 20) << 20) | (o & 0x0fffff);
    }

    inline bool operator < (const MultiNote& rhs) const {
        if ((shapeIndexAndOnset & 0x0fffff) == rhs.getOnset()) {
            return pitch < rhs.pitch;
        }
        return (shapeIndexAndOnset & 0x0fffff) < rhs.getOnset();
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

void printTrack(const vector<MultiNote>& track, const vector<Shape> shapeDict, const size_t begin, const size_t length) {
    for (int i = begin; i < begin + length; ++i) {
        cout << i << " - Shape=" << shape2String(shapeDict[track[i].getShapeIndex()]);
        cout << " onset=" << (int) track[i].getOnset()
                << " basePitch=" << (int) track[i].pitch
                << " timeUnit=" << (int) track[i].unit
                << " velocity=" << (int) track[i].vel;
        cout << " neighbor=" << (int) track[i].neighbor << endl;
    }
}

void updateNeighbor(vector<Piece>& corpus, const vector<Shape> shapeDict, long maxDurInNth) {
    // for each piece
    #pragma omp parallel for
    for (int i = 0; i < corpus.size(); ++i) {
        // for each track
        #pragma omp parallel for
        for (int j = 0; j < corpus[i].size(); ++j) {
            // for each multinote
            for (int k = 0; k < corpus[i][j].size(); ++k) {
                // printTrack(corpus[i][j], k, 1);
                uint32_t onsetTime = corpus[i][j][k].getOnset();
                RelNote latestRelNote = shapeDict[corpus[i][j][k].getShapeIndex()].back();
                uint32_t relOffsetTime = latestRelNote.getRelOnset() + latestRelNote.relDur;
                uint32_t offsetTime = corpus[i][j][k].getOnset() + relOffsetTime * corpus[i][j][k].unit;
                uint32_t immdAfterOnset = -1;
                int n = 1;
                corpus[i][j][k].neighbor = 0;
                while (k+n < corpus[i][j].size()) {
                    // not allow this because it has possibility to make timeUnit > maxDurInNth
                    if ((corpus[i][j][k+n]).getOnset() > onsetTime + maxDurInNth) {
                        break;
                    }
                    // is overlapping
                    if ((corpus[i][j][k+n]).getOnset() < offsetTime) { 
                        n++;
                    }
                    // if immediately after
                    else if (immdAfterOnset == -1 || (corpus[i][j][k+n]).getOnset() == immdAfterOnset) {
                        immdAfterOnset = (corpus[i][j][k+n]).getOnset();
                        n++;
                    }
                    else {
                        break;
                    }
                }
                corpus[i][j][k].neighbor = n - 1;
            }
        }
    }
}

Shape getShapeOfMultiNotePair(const MultiNote& lmn, const MultiNote& rmn, const Shape& lShape, const Shape& rShape) {
    int leftSize = lShape.size(), rightSize = rShape.size();
    int pairSize = leftSize + rightSize;
    Shape pairShape;
    pairShape.resize(pairSize);

    int rightOnsetsDurs[rightSize*2];
    for (int i = 0; i < rightSize; ++i) {
        rightOnsetsDurs[i] = rShape[i].getRelOnset() * rmn.unit + rmn.getOnset() - lmn.getOnset();
    }
     for (int i = 0; i < rightSize; ++i) {
        rightOnsetsDurs[i+rightSize] = rShape[i].relDur * rmn.unit;
    }
    int newUnit = gcd(lmn.unit, gcd(rightOnsetsDurs, rightSize*2));
    // cout << "newUnit:" << newUnit << endl;
    for (int i = 0; i < pairSize; ++i) {
        if (i < leftSize) {
            if (i != 0) {
                pairShape[i].setRelOnset(lShape[i].getRelOnset() * lmn.unit / newUnit);
                pairShape[i].relPitch = lShape[i].relPitch;
            }
            pairShape[i].relDur = lShape[i].relDur * lmn.unit / newUnit;
            pairShape[i].setCont(lShape[i].isCont());
        }
        else {
            int j = i - leftSize;
            pairShape[i].setRelOnset(rightOnsetsDurs[j] / newUnit);
            pairShape[i].relPitch = rShape[j].relPitch + rmn.pitch - lmn.pitch;
            pairShape[i].relDur = rightOnsetsDurs[j+rightSize] / newUnit;
            pairShape[i].setCont(rShape[j].isCont());
        }
    }
    sort(pairShape.begin(), pairShape.end());
    return pairShape;
}


int main(int argc, char *argv[]) {
    // validate args
    if (argc != 3) {
        cout << "Must have 2 arguments: [bpeIter] [corpusFilePath]" << endl;
        return 1;
    }
    int bpeIter = atoi(argv[1]);
    if (bpeIter) {
        if (bpeIter > 2045) {
            cout << "bpeIter can not be greater than 2045: " << bpeIter << endl;
        }
        cout << "bpeIter: " << bpeIter << endl;
    }
    else {
        cout << "Third arguments [bpeIter] is not convertable by strtol in base 10: " << argv[1] << endl;
        return 1;
    }

    ifstream infile(argv[2], ios::in | ios::binary);
    if (!infile.is_open()) {
        cout << "Failed to open corpus file: " << argv[2] << endl;
        return 1;
    }
    cout << "Input file: " << argv[2] << endl;

    string vocabOutfileName(argv[2]);
    vocabOutfileName.append("_vocabs");
    ofstream vocabOutfile(vocabOutfileName, ios::out | ios::trunc);
    if (!vocabOutfile.is_open()) {
        cout << "Failed to open vocab output file: " << vocabOutfileName << endl;
        return 1;
    }

    string textOutfileName(argv[2]);
    textOutfileName.append("_multinotes");
    ofstream textOutfile(textOutfileName, ios::out | ios::trunc);
    if (!textOutfile.is_open()) {
        cout << "Failed to open text output file: " << textOutfileName << endl;
        return 1;
    }

    time_t begTime = time(0);

    // read & construct data
    infile.seekg(0, ios::end);
    streampos filesize = infile.tellg();
    infile.seekg(0, ios::beg);
    cout << "File size: " << filesize << " bytes" << endl;

    // read notes from corpus
    vector<Piece> corpus;
    corpus.push_back(Piece());

    // ignore yaml head matters
    string yaml;
    int nth = 0, maxDurInNth = 0, maxDurInBeat = 0;
    while (1) {
        getline(infile, yaml);
        if (yaml.substr(0, 3) == "---") {
            break;
        }
        else if (yaml.substr(0, 3) == "nth") {
            nth = atoi(yaml.substr(5).c_str());
        }
        else if (yaml.substr(0, 12) == "max_duration") {
            maxDurInBeat = atoi(yaml.substr(14).c_str());
        }
    }
    maxDurInNth = nth / 4 * maxDurInBeat;
    if (!nth || !maxDurInNth) {
        cout << "nth or maxDurInNth is zero: nth=" << nth << " maxDurInNth=" << maxDurInNth << endl;
        return 1;
    }
    else {
        cout << "nth: " << nth << endl;
        cout << "maxDurInNth: " << maxDurInNth << endl;
    }

    unsigned int curMeasureStart = 0, curMeasureLength = 0, curTime = 0;
    while (infile.good()) {
        unsigned char c = infile.get(), i;
        char a[8];
        switch (c) {
            case 'R':
                corpus.back().push_back(vector<MultiNote>());
                while (infile.get() != ' ');
                break;
            case 'M':
                curMeasureStart += curMeasureLength;
                uint8_t numer, denom;
                for (i = 0; (a[i] = infile.get()) != '/'; ++i);
                a[i] = '\0';
                numer = b36strtol(a);
                // n = strtol(a, &pEnd, 36);
                for (i = 0; (a[i] = infile.get()) != ' '; ++i);
                a[i] = '\0';
                denom = b36strtol(a);
                // d = strtol(a, &pEnd, 36);
                curMeasureLength = numer * nth / denom;
                break;
            case 'P':
                uint8_t pos;
                for (i = 0; (a[i] = infile.get()) != ' '; ++i);
                a[i] = '\0';
                pos = b36strtol(a);
                // p = strtol(a, &pEnd, 36);
                curTime = curMeasureStart + pos;
                break;
            case 'N':
                uint8_t isCont, p, d, v, t;
                if (isCont = (infile.get() == '~')) {
                    infile.get();
                }
                for (i = 0; (a[i] = infile.get()) != ':'; ++i);
                a[i] = '\0';
                p = b36strtol(a);
                // p = strtol(a, &pEnd, 36);
                for (i = 0; (a[i] = infile.get()) != ':'; ++i);
                a[i] = '\0';
                d = b36strtol(a);
                for (i = 0; (a[i] = infile.get()) != ':'; ++i);
                a[i] = '\0';
                v = b36strtol(a);
                for (i = 0; (a[i] = infile.get()) != ' '; ++i);
                a[i] = '\0';
                t = b36strtol(a);
                corpus.back()[t].push_back(MultiNote(isCont, curTime, p, d, v));
                break;
            case '\n':
                corpus.push_back(Piece());
                curMeasureStart = 0; curMeasureLength = 0; curTime = 0;
                break;
            default:
                break;
        }
    }
    if (corpus.back().size() == 0) {
        corpus.pop_back();
    }
    infile.close();

    vector<Shape> shapeDict;
    shapeDict.reserve(bpeIter + 2);
    shapeDict.push_back({RelNote(0, 0, 0, 1)}); // DEFAULT_SHAPE_END
    shapeDict.push_back({RelNote(1, 0, 0, 1)}); // DEFAULT_SHAPE_CONT

    // sort and count notes
    int maxTrackNum = 0;
    size_t multinoteCount = 0;
    #pragma omp parallel for
    for (unsigned int i = 0; i < corpus.size(); ++i) {
        if (maxTrackNum < corpus[i].size()) {
            maxTrackNum = corpus[i].size();
        }
        for (unsigned int j = 0; j < corpus[i].size(); ++j) {
            multinoteCount += corpus[i][j].size();
            sort(corpus[i][j].begin(), corpus[i][j].end());
        }
    }
    // printTrack(corpus[0][0], shapeDict, 0, 10);
    cout << "Start multinote count: " << multinoteCount << " Time: " << (unsigned int) time(0) - begTime << endl;
    if (multinoteCount == 0) {
        cout << "Exit for zero notes" << endl;
        return 1;
    }

    map<Shape, unsigned int> shapeOccurCount;
    map<Shape, unsigned int> shapeOccurCountParallel[8];

    for (int iterCount = 0; iterCount < bpeIter; ++iterCount) {
        cout << "iter:" << iterCount << ", ";
        begTime = time(0);

        updateNeighbor(corpus, shapeDict, maxDurInNth);

        // count shape frequency
        // for each piece
        for (int i = 0; i < corpus.size(); ++i) {
            // for each track
            #pragma omp parallel for num_threads(8)
            for (int j = 0; j < corpus[i].size(); ++j) {
                // for each multinote
                int thread_num = omp_get_thread_num();
                // int thread_num = j % 8;
                Shape s;
                for (int k = 0; k < corpus[i][j].size(); ++k) {
                    // for each neighbor
                    for (int n = 1; n < corpus[i][j][k].neighbor; ++n) {
                        if (corpus[i][j][k].vel != corpus[i][j][k+n].vel) continue;
                        // if (DEBUG) cout << i << "," << j << ":" << k << "->" << k+n;

                        s = getShapeOfMultiNotePair(
                            corpus[i][j][k],
                            corpus[i][j][k+n],
                            shapeDict[corpus[i][j][k].getShapeIndex()],
                            shapeDict[corpus[i][j][k+n].getShapeIndex()]
                        );

                        if (shapeOccurCountParallel[thread_num].count(s) == 0) {
                            // if (DEBUG) cout << " " << shape2String(s) << " not in map" << endl;
                            shapeOccurCountParallel[thread_num].insert(pair<Shape, unsigned int>(s, (unsigned int) 1));
                        }
                        else {
                            // if (DEBUG) cout << " " << shape2String(s) << " in map" << endl;
                            shapeOccurCountParallel[thread_num][s]++;
                        }
                    }
                }
            }
        }

        // cout << "merging occur count" << endl;
        // merge parrallel maps
        for (int j = 0; j < 8; ++j) {
            if (j == 0) {
                for (auto it = shapeOccurCountParallel[j].begin(); it != shapeOccurCountParallel[j].end(); it++) {
                    shapeOccurCount.insert(*it);
                }
            }
            else {
                for (auto it = shapeOccurCountParallel[j].begin(); it != shapeOccurCountParallel[j].end(); it++) {
                    if (shapeOccurCount.count(it->first)) {
                        shapeOccurCount[it->first] += it->second;
                    }
                    else {
                        shapeOccurCount.insert(*it);
                    }
                }
            }
            shapeOccurCountParallel[j].clear();
        }
        cout << "Find " << shapeOccurCount.size() << " unique pairs" << ", ";
    
        // add shape with highest frequency into shapeDict
        Shape maxFreqShape;
        unsigned int maxFreq = 0;
        for (auto it = shapeOccurCount.cbegin(); it != shapeOccurCount.cend(); it++) {
            if (maxFreq < (*it).second) {
                maxFreqShape = (*it).first;
                maxFreq = (*it).second;
            }
            // cout << shape2String((*it).first) << endl;
        }

        unsigned int newShapeIndex = shapeDict.size();
        shapeDict.push_back(maxFreqShape);
        cout << "Add new shape: " << shape2String(maxFreqShape) << " freq=" << maxFreq << ", ";

        // merge MultiNotes with new added shape
        // for each piece
        #pragma omp parallel for
        for (int i = 0; i < corpus.size(); ++i) {
            // for each track
            #pragma omp parallel for
            for (int j = 0; j < corpus[i].size(); ++j) {
                // for each multinote
                // iterate backward to preseve the index relationship for neighbors as much as possible
                for (int k = corpus[i][j].size()-1; k >= 0; --k) {
                    // for each neighbor
                    for (int n = 1; n < corpus[i][j][k].neighbor; ++n) {
                        if (k + n >= corpus[i][j].size()) {
                            continue;
                        }
                        if (corpus[i][j][k].vel != corpus[i][j][k+n].vel) {
                            continue;
                        }
                        if (shapeDict[corpus[i][j][k].getShapeIndex()].size()
                          + shapeDict[corpus[i][j][k+n].getShapeIndex()].size()
                          != maxFreqShape.size()) {
                            continue;
                        }
                        Shape s = getShapeOfMultiNotePair(
                            corpus[i][j][k],
                            corpus[i][j][k+n],
                            shapeDict[corpus[i][j][k].getShapeIndex()],
                            shapeDict[corpus[i][j][k+n].getShapeIndex()]
                        );
                        if (s == maxFreqShape) {
                            // if (DEBUG) {
                            //     cout << "occur: " << i << ',' << j << ':' << k << '-' << k+n << endl;
                            //     printTrack(corpus[i][j], k, 1);
                            //     printTrack(corpus[i][j], k+n, 1);
                            // }

                            // change left multinote to merged multinote
                            corpus[i][j][k].unit = shapeDict[corpus[i][j][k].getShapeIndex()][0].relDur * corpus[i][j][k].unit / maxFreqShape[0].relDur;
                            corpus[i][j][k].setShapeIndex(newShapeIndex);

                            // if (DEBUG) {
                            //     cout << "NEW: "; printTrack(corpus[i][j], k, 1);
                            //     cout << "--------\n";
                            // }

                            // remove right multinote
                            // it is ok to do it like this since we'll sort it later
                            corpus[i][j][k+n] = corpus[i][j].back();
                            corpus[i][j].pop_back();
                            break;
                        }
                    }
                }
                sort(corpus[i][j].begin(), corpus[i][j].end());
            }
        }
        
        shapeOccurCount.clear();
        cout << "Corpus updated. ";
        cout << "Time: " << (unsigned int) time(0) - begTime;
        cout << endl;
    }

    multinoteCount = 0;
    #pragma omp parallel for reduction(+:multinoteCount)
    for (int i = 0; i < corpus.size(); ++i) {
        for (int j = 0; j < corpus[i].size(); ++j) {
            multinoteCount += corpus[i][j].size();
        }
    }
    cout << "Final multinote count: " << multinoteCount << endl;

    // write vocab file
    // wont write the first 2 default shape
    for (int i = 2; i < shapeDict.size(); ++i) {
        stringstream ss;
        ss << 'S' << shape2String(shapeDict[i]);
        vocabOutfile << ss.str() << endl;
    }
    cout << "Write vocabs file done." << endl;

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
                if (minOnset > corpus[i][j][curIndex[j]].getOnset()) {
                    minOnset = corpus[i][j][curIndex[j]].getOnset();
                    minOnsetTrack = j;
                }
            }
            unsigned int curShapeIndex = corpus[i][minOnsetTrack][curIndex[minOnsetTrack]].getShapeIndex();
            string shapeStr;
            if (curShapeIndex == 0) {
                shapeStr = "N";
            }
            else if (curShapeIndex == 1) {
                shapeStr = "N~";
            }
            else {
                shapeStr = 'S' + ltob36str(curShapeIndex-2);
            }
            stringstream s;
            s << ltob36str(corpus[i][minOnsetTrack][curIndex[minOnsetTrack]].getOnset()) << '@'
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
    cout << "Write multinotes text file done." << endl;

    return 0;
}