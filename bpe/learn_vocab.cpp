#include "multinotes.hpp" // iostream in there
#include "corpusIO.hpp"
#include <cstring>
#include <ctime>
#include <algorithm>
#include <vector>
#include <list>
#include "omp.h"

#define COUNTING_THREAD_NUM 8

using namespace std;

int gcd (int a, int b) {
    if (a == b) return a;
    if (a == 0 || b == 0) return ((a > b) ? a : b);
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
        if (arr[i] != 0) {
            g = gcd(g, arr[i]);
        }
        if (g == 1) return 1;
    }
    return g;
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

unsigned int findMaxRelOffset(const Shape& s) {
    unsigned int maxRelOffset = s[0].getRelOnset() + s[0].relDur;
    for (int i = 1; i < s.size(); ++i) {
        if (maxRelOffset < s[i].getRelOnset() + s[i].relDur) {
            maxRelOffset = s[i].getRelOnset() + s[i].relDur;
        }
    }
    return maxRelOffset;
}

void updateNeighbor(Corpus& corpus, const vector<Shape>& shapeDict, long maxDur) {
    // for each piece
    #pragma omp parallel for
    for (int i = 0; i < corpus.piecesMN.size(); ++i) {
        // for each track
        #pragma omp parallel for
        for (int j = 0; j < corpus.piecesMN[i].size(); ++j) {
            if (corpus.piecesTN[i][j] != 128) {
                // is not drum
                // for each multinote
                for (int k = 0; k < corpus.piecesMN[i][j].size(); ++k) {
                    // printTrack(corpus[i][j], k, 1);
                    uint32_t onsetTime = corpus.piecesMN[i][j][k].getOnset();
                    uint32_t maxRelOffset = findMaxRelOffset(shapeDict[corpus.piecesMN[i][j][k].getShapeIndex()]);
                    uint32_t offsetTime = corpus.piecesMN[i][j][k].getOnset() + maxRelOffset * corpus.piecesMN[i][j][k].unit;
                    uint32_t immdAfterOnset = -1;
                    int n = 1;
                    corpus.piecesMN[i][j][k].neighbor = 0;
                    while (k+n < corpus.piecesMN[i][j].size()) {
                        // not allow this because it has possibility to make timeUnit > maxDur
                        if ((corpus.piecesMN[i][j][k+n]).getOnset() > onsetTime + maxDur) {
                            break;
                        }
                        // overlapping
                        if ((corpus.piecesMN[i][j][k+n]).getOnset() < offsetTime) { 
                            n++;
                        }
                        // immediately after
                        else if (immdAfterOnset == -1 || (corpus.piecesMN[i][j][k+n]).getOnset() == immdAfterOnset) {
                            immdAfterOnset = (corpus.piecesMN[i][j][k+n]).getOnset();
                            n++;
                        }
                        else {
                            break;
                        }
                    }
                    corpus.piecesMN[i][j][k].neighbor = n - 1;
                }
            }
        }
    }
}

Shape getShapeOfMultiNotePair(const MultiNote& lmn, const MultiNote& rmn, const Shape& lShape, const Shape& rShape) {
    int leftSize = lShape.size(), rightSize = rShape.size();
    int pairSize = leftSize + rightSize;
    Shape pairShape;
    pairShape.resize(pairSize);

    int rightRelOnsets[rightSize];
    for (int i = 0; i < rightSize; ++i) {
        rightRelOnsets[i] = rShape[i].getRelOnset() * rmn.unit + rmn.getOnset() - lmn.getOnset();
    }
    int newUnit = gcd(gcd(lmn.unit, rmn.unit), gcd(rightRelOnsets, rightSize));
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
            pairShape[i].setRelOnset(rightRelOnsets[j] / newUnit);
            pairShape[i].relPitch = rShape[j].relPitch + rmn.pitch - lmn.pitch;
            pairShape[i].relDur = rShape[j].relDur * rmn.unit / newUnit;
            pairShape[i].setCont(rShape[j].isCont());
        }
    }
    sort(pairShape.begin(), pairShape.end());
    pairShape.shrink_to_fit();
    return pairShape;
}


int main(int argc, char *argv[]) {
    // read and validate args
    if (argc != 3) {
        cout << "Must have 2 arguments: [bpeIter] [corpusDirPath]" << endl;
        return 1;
    }
    int bpeIter = atoi(argv[1]);
    if (bpeIter) {
        if (bpeIter > 2045) {
            cout << "bpeIter can not be greater than 2045: " << bpeIter << endl;
            return 1;
        }
        cout << "bpeIter: " << bpeIter << endl;
    }
    else {
        cout << "Third arguments [bpeIter] is not convertable by atoi: " << argv[1] << endl;
        return 1;
    }
    string corpusDirPath(argv[2]);

    // open files
    string corpusFilePath = corpusDirPath + "/corpus";
    cout << "Input corpus file path: " << corpusFilePath << endl;
    ifstream corpusFile(corpusFilePath, ios::in | ios::binary);
    if (!corpusFile.is_open()) {
        cout << "Failed to open corpus file: " << corpusFilePath << endl;
        return 1;
    }
    
    string parasFilePath = corpusDirPath + "/paras";
    cout << "Input parameter file path: " << corpusFilePath << endl;
    ifstream parasFile(parasFilePath, ios::in | ios::binary);
    if (!parasFile.is_open()) {
        cout << "Failed to open parameters file: " << parasFilePath << endl;
        return 1;
    }

    string vocabOutFilePath = corpusDirPath + "_bpeiter" + to_string(bpeIter) + "/shape_vocab";
    cout << "Output shape vocab file path: " << vocabOutFilePath << endl;
    ofstream vocabOutfile(vocabOutFilePath, ios::out | ios::trunc);
    if (!vocabOutfile.is_open()) {
        cout << "Failed to open vocab output file: " << vocabOutFilePath << endl;
        return 1;
    }

    string tokenizedCorpusFilePath = corpusDirPath + "_bpeiter" + to_string(bpeIter) + "/corpus";
    cout << "Output tokenized corpus file path: " << tokenizedCorpusFilePath << endl;
    ofstream tokenizedCorpusFile(tokenizedCorpusFilePath, ios::out | ios::trunc);
    if (!tokenizedCorpusFile.is_open()) {
        cout << "Failed to open tokenized corpus output file: " << tokenizedCorpusFilePath << endl;
        return 1;
    }

    time_t begTime = time(0);

    // read parameters
    map<string, string> paras = readParasFile(parasFile);
    int nth, maxDur, maxTrackNum;
    string positionMethod;
    // stoi, c++11 thing
    nth = stoi(paras[string("nth")]);
    maxDur = stoi(paras[string("max_duration")]);
    maxTrackNum = stoi(paras[string("max_track_number")]);
    positionMethod = paras[string("position_method")];
    cout << "nth=" << nth <<  endl
        << "maxDur=" << maxDur << endl
        << "maxTrackNum" << maxTrackNum << endl
        << "positionMethod=" << positionMethod << endl;
    if (!nth || !maxDur || maxDur > 255 || !maxTrackNum || (positionMethod != "event" || positionMethod != "attribute")) {
        cout << "Corpus parameter errror" << endl;
        return 1;
    }

    // read notes from corpus
    Corpus corpus = readCorpusFile(corpusFile, nth, positionMethod);

    vector<Shape> shapeDict;
    shapeDict.reserve(bpeIter + 2);
    shapeDict.push_back({RelNote(0, 0, 0, 1)}); // DEFAULT_SHAPE_END
    shapeDict.push_back({RelNote(1, 0, 0, 1)}); // DEFAULT_SHAPE_CONT

    // sort and count notes
    size_t multinoteCount = 0;
    for (unsigned int i = 0; i < corpus.piecesMN.size(); ++i) {
        #pragma omp parallel for reduction(+:multinoteCount)
        for (unsigned int j = 0; j < corpus.piecesMN[i].size(); ++j) {
            multinoteCount += corpus.piecesMN[i][j].size();
            sort(corpus.piecesMN[i][j].begin(), corpus.piecesMN[i][j].end());
        }
    }
    // printTrack(corpus[0][0], shapeDict, 0, 10);
    cout << "Beginning multinote count: " << multinoteCount << ", Time: " << (unsigned int) time(0) - begTime << endl;
    if (multinoteCount == 0) {
        cout << "Exit for zero notes" << endl;
        return 1;
    }

    map<Shape, unsigned int> shapeOcorpusurCount;
    map<Shape, unsigned int> shapeOcorpusurCountParallel[COUNTING_THREAD_NUM];

    for (int iterCount = 0; iterCount < bpeIter; ++iterCount) {
        cout << "Iter:" << iterCount << ", ";
        begTime = time(0);

        updateNeighbor(corpus, shapeDict, maxDur);

        // count shape frequency
        // for each piece
        #pragma omp parallel for num_threads(COUNTING_THREAD_NUM)
        for (int i = 0; i < corpus.size(); ++i) {
            // for each track
            for (int j = 0; j < corpus[i].size(); ++j) {
                // for each multinote
                int thread_num = omp_get_thread_num();
                // int thread_num = j % COUNTING_THREAD_NUM;
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

                        if (shapeOcorpusurCountParallel[thread_num].count(s) == 0) {
                            // if (DEBUG) cout << " " << shape2String(s) << " not in map" << endl;
                            shapeOcorpusurCountParallel[thread_num].insert(pair<Shape, unsigned int>(s, (unsigned int) 1));
                        }
                        else {
                            // if (DEBUG) cout << " " << shape2String(s) << " in map" << endl;
                            shapeOcorpusurCountParallel[thread_num][s]++;
                        }
                    }
                }
            }
        }

        // cout << "merging ocorpusur count" << endl;
        // merge parrallel maps
        for (int j = 0; j < 8; ++j) {
            if (j == 0) {
                for (auto it = shapeOcorpusurCountParallel[j].begin(); it != shapeOcorpusurCountParallel[j].end(); it++) {
                    shapeOcorpusurCount.insert(*it);
                }
            }
            else {
                for (auto it = shapeOcorpusurCountParallel[j].begin(); it != shapeOcorpusurCountParallel[j].end(); it++) {
                    if (shapeOcorpusurCount.count(it->first)) {
                        shapeOcorpusurCount[it->first] += it->second;
                    }
                    else {
                        shapeOcorpusurCount.insert(*it);
                    }
                }
            }
            shapeOcorpusurCountParallel[j].clear();
        }
        cout << "Find " << shapeOcorpusurCount.size() << " unique pairs" << ", ";
    
        // add shape with highest frequency into shapeDict
        Shape maxFreqShape;
        unsigned int maxFreq = 0;
        for (auto it = shapeOcorpusurCount.cbegin(); it != shapeOcorpusurCount.cend(); it++) {
            if (maxFreq < (*it).second) {
                maxFreqShape = (*it).first;
                maxFreq = (*it).second;
            }
            // cout << shape2String((*it).first) << endl;
        }

        unsigned int newShapeIndex = shapeDict.size();
        shapeDict.push_back(maxFreqShape);
        cout << "Add new shape: " << shape2String(maxFreqShape) << " freq=" << maxFreq << endl;

        // merge MultiNotes with new added shape
        // for each piece
        #pragma omp parallel for
        for (int i = 0; i < corpus.size(); ++i) {
            // for each track
            #pragma omp parallel for
            for (int j = 0; j < corpus[i].size(); ++j) {
                // for each multinote
                // iterate forward
                for (int k = 0; k < corpus[i][j].size(); ++k) {
                    // for each neighbor
                    for (int n = 1; n < corpus[i][j][k].neighbor; ++n) {
                        if (k + n >= corpus[i][j].size()) {
                            continue;
                        }
                        if (corpus[i][j][k].vel != corpus[i][j][k+n].vel
                           || corpus[i][j][k].vel == 0
                           || corpus[i][j][k+n].vel == 0) {
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
                            // change left multinote to merged multinote
                            // because the relnotes are sorted in same way as multinotes,
                            // we can prove that the first relnote in the new shape is correspond to the first relnote in left multinote's original shape
                            corpus[i][j][k].unit = shapeDict[corpus[i][j][k].getShapeIndex()][0].relDur * corpus[i][j][k].unit / maxFreqShape[0].relDur;
                            corpus[i][j][k].setShapeIndex(newShapeIndex);

                            // mark right multinote to be removed by have vel set to 0
                            corpus[i][j][k+n].vel = 0;
                            break;
                        }
                    }
                }
                // "delete" multinotes with vel == 0
                // to prevent that back() is also to be removed, we iterate from back to front
                for (int k = corpus[i][j].size() - 1; k >= 0; --k) {
                    if (corpus[i][j][k].vel == 0) {
                        corpus[i][j][k] = corpus[i][j].back();
                        corpus[i][j].pop_back();
                    }
                }
                sort(corpus[i][j].begin(), corpus[i][j].end());
            }
        }

        multinoteCount = 0;
        #pragma omp parallel for reduction(+:multinoteCount)
        for (int i = 0; i < corpus.size(); ++i) {
            for (int j = 0; j < corpus[i].size(); ++j) {
                // cout << i << ',' << j << " count: " << corpus[i][j].size() << endl;
                // sort(corpus[i][j].begin(), corpus[i][j].end());
                multinoteCount += corpus[i][j].size();
            }
        }
        cout << "After merged, multinote count: " << multinoteCount << ", ";
        
        shapeOcorpusurCount.clear();
        // cout << "Corpus updated. ";
        cout << "Time: " << (unsigned int) time(0) - begTime;
        cout << endl;
    }

    // write vocab file
    // wont write the first 2 default shape
    for (int i = 2; i < shapeDict.size(); ++i) {
        stringstream ss;
        ss << 'S' << shape2String(shapeDict[i]);
        vocabOutfile << ss.str() << endl;
    }
    cout << "Write vocabs file done." << endl;

    begTime = time(0);

    corpusFile.clear(); // have to clear because we reached eof at the begining
    corpusFile.seekg(0, ios::beg);

    

    cout << "Write tokenized corpus file done. Time:" << time(0)-begTime << endl;
    return 0;
}