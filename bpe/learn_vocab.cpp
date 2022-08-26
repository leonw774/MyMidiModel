#include "corpus.hpp"
#include "shapecounting.hpp"
#include <string>
#include <algorithm>
#include <ctime>

using namespace std;

void printTrack(const Track& track, const vector<Shape>& shapeDict, const size_t begin, const size_t length) {
    for (int i = begin; i < begin + length; ++i) {
        cout << i << " - Shape=" << shape2str(shapeDict[track[i].getShapeIndex()]);
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
    // #pragma omp parallel for
    for (int i = 0; i < corpus.piecesMN.size(); ++i) {
        // for each track
        // #pragma omp parallel for
        for (int j = 0; j < corpus.piecesMN[i].size(); ++j) {
            // ignore drum
            if (corpus.piecesTN[i][j] == 128) continue;
            // for each multinote
            for (int k = 0; k < corpus.piecesMN[i][j].size(); ++k) {
                // printTrack(corpus.piecesMN[i][j], shapeDict, k, 1);
                uint32_t onsetTime = corpus.piecesMN[i][j][k].getOnset();
                uint32_t maxRelOffset = findMaxRelOffset(shapeDict[corpus.piecesMN[i][j][k].getShapeIndex()]);
                uint32_t offsetTime = corpus.piecesMN[i][j][k].getOnset() + maxRelOffset * corpus.piecesMN[i][j][k].unit;
                uint32_t immdAfterOnset = -1;
                int n = 1;
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

int main(int argc, char *argv[]) {
    // read and validate args
    if (argc != 6) {
        cout << "Must have 5 arguments: [corpusDirPath] [bpeIter] [samplingRate] [alpha] [prfTopK (0=not use)]" << endl;
        return 1;
    }
    string corpusDirPath(argv[1]);
    int bpeIter = atoi(argv[2]);
    double samplingRate = atof(argv[3]);
    double alpha = atof(argv[4]);
    bool prfTopK = atoi(argv[5]);
    if (bpeIter <= 0 || 2045 < bpeIter) {
        cout << "Error: bpeIter <= 0 or > 2045: " << bpeIter << endl;
        return 1;
    }
    if (prfTopK < 0) {
        cout << "Error: prfTopK < 0" << endl;
    }
    cout << "corpusDirPath: " << corpusDirPath << endl
        << "bpeIter: " << bpeIter << endl
        << "samplingRate: " << samplingRate << endl
        << "alpha: " << alpha << endl
        << "prfTopK: " << prfTopK << endl;

    // open files
    string corpusFilePath = corpusDirPath + "/corpus";
    cout << "Input corpus file path: " << corpusFilePath << endl;
    ifstream corpusFile(corpusFilePath, ios::in | ios::binary);
    if (!corpusFile.is_open()) {
        cout << "Failed to open corpus file: " << corpusFilePath << endl;
        return 1;
    }
    
    string parasFilePath = corpusDirPath + "/paras";
    cout << "Input parameter file path: " << parasFilePath << endl;
    ifstream parasFile(parasFilePath, ios::in | ios::binary);
    if (!parasFile.is_open()) {
        cout << "Failed to open parameters file: " << parasFilePath << endl;
        return 1;
    }

    string vocabFilePath = corpusDirPath + "_bpeiter" + to_string(bpeIter) + "/shape_vocab";
    cout << "Output shape vocab file path: " << vocabFilePath << endl;
    ofstream vocabFile(vocabFilePath, ios::out | ios::trunc);
    if (!vocabFile.is_open()) {
        cout << "Failed to open vocab output file: " << vocabFilePath << endl;
        return 1;
    }

    string outputCorpusFilePath = corpusDirPath + "_bpeiter" + to_string(bpeIter) + "/corpus";
    cout << "Output merged corpus file path: " << outputCorpusFilePath << endl;
    ofstream outputCorpusFile(outputCorpusFilePath, ios::out | ios::trunc);
    if (!outputCorpusFile.is_open()) {
        cout << "Failed to open tokenized corpus output file: " << outputCorpusFilePath << endl;
        return 1;
    }

    time_t begTime = time(0);

    // read parameters
    map<string, string> paras = readParasFile(parasFile);
    int nth, maxDur, maxTrackNum;
    string positionMethod;
    // stoi, c++11 thing
    nth = stoi(paras[string("nth")]);
    cout << "nth: " << nth << endl;
    maxDur = stoi(paras[string("max_duration")]);
    cout << "maxDur: " << maxDur << endl;
    maxTrackNum = stoi(paras[string("max_track_number")]);
    cout << "maxTrackNum: " << maxTrackNum << endl;
    positionMethod = paras[string("position_method")];
    cout << "positionMethod=" << positionMethod << endl;
    if (nth <= 0 || maxDur <= 0 || maxDur > 255 || maxTrackNum <= 0 || (positionMethod != "event" && positionMethod != "attribute")) {
        cout << "Corpus parameter error" << endl;
        return 1;
    }

    // read notes from corpus
    cout << "Reading corpus" << endl;
    Corpus corpus = readCorpusFile(corpusFile, nth, positionMethod);
    cout << "Read corpus complete" << endl;

    vector<Shape> shapeDict;
    shapeDict.reserve(bpeIter + 2);
    shapeDict.push_back({RelNote(0, 0, 0, 1)}); // DEFAULT_SHAPE_END
    shapeDict.push_back({RelNote(1, 0, 0, 1)}); // DEFAULT_SHAPE_CONT

    // sort and count notes
    size_t multinoteCount = 0;
    for (int i = 0; i < corpus.piecesMN.size(); ++i) {
        #pragma omp parallel for reduction(+:multinoteCount)
        for (int j = 0; j < corpus.piecesMN[i].size(); ++j) {
            multinoteCount += corpus.piecesMN[i][j].size();
            sort(corpus.piecesMN[i][j].begin(), corpus.piecesMN[i][j].end());
        }
    }
    // printTrack(corpus[0][0], shapeDict, 0, 10);
    cout << "Multinote count at start: " << multinoteCount << ", Time: " << (unsigned int) time(0) - begTime << endl;
    if (multinoteCount == 0) {
        cout << "Exit for zero notes" << endl;
        return 1;
    }

    time_t iterTime;
    map<Shape, int> shapeCount;
    for (int iterCount = 0; iterCount < bpeIter; ++iterCount) {
        cout << "Iter:" << iterCount << ", ";
        iterTime = time(0);

        updateNeighbor(corpus, shapeDict, maxDur);

        // count shape frequency
        if (prfTopK) {
            pseudoRelevanceFeedbackInspiredShapeCounting(
                corpus,
                shapeDict,
                shapeCount,
                prfTopK,
                samplingRate,
                alpha
            );
        }
        else {
            basicShapeCounting(
                corpus,
                shapeDict,
                shapeCount,
                samplingRate,
                alpha
            );
        }
        cout << "Find " << shapeCount.size() << " unique pairs" << ", ";
    
        // add shape with highest frequency into shapeDict
        Shape maxScoreShape;
        int maxScore = 0;
        for (auto it = shapeCount.cbegin(); it != shapeCount.cend(); it++) {
            if (maxScore < (*it).second) {
                maxScoreShape = (*it).first;
                maxScore = (*it).second;
            }
            // cout << shape2str((*it).first) << endl;
        }

        unsigned int newShapeIndex = shapeDict.size();
        shapeDict.push_back(maxScoreShape);
        cout << "New shape: " << shape2str(maxScoreShape) << " Score=" << maxScore << endl;

        // merge MultiNotes with newly added shape
        // for each piece
        #pragma omp parallel for
        for (int i = 0; i < corpus.piecesMN.size(); ++i) {
            // for each track
            #pragma omp parallel for
            for (int j = 0; j < corpus.piecesMN[i].size(); ++j) {
                // ignore drum
                if (corpus.piecesTN[i][j] == 128) continue;
                // for each multinote
                // iterate forward
                for (int k = 0; k < corpus.piecesMN[i][j].size(); ++k) {
                    // for each neighbor
                    for (int n = 1; n < corpus.piecesMN[i][j][k].neighbor; ++n) {
                        if (k + n >= corpus.piecesMN[i][j].size()) {
                            continue;
                        }
                        if (corpus.piecesMN[i][j][k].vel != corpus.piecesMN[i][j][k+n].vel
                           || corpus.piecesMN[i][j][k].vel == 0
                           || corpus.piecesMN[i][j][k+n].vel == 0) {
                            continue;
                        }
                        if (shapeDict[corpus.piecesMN[i][j][k].getShapeIndex()].size()
                          + shapeDict[corpus.piecesMN[i][j][k+n].getShapeIndex()].size()
                          != maxScoreShape.size()) {
                            continue;
                        }
                        Shape s = getShapeOfMultiNotePair(
                            corpus.piecesMN[i][j][k],
                            corpus.piecesMN[i][j][k+n],
                            shapeDict[corpus.piecesMN[i][j][k].getShapeIndex()],
                            shapeDict[corpus.piecesMN[i][j][k+n].getShapeIndex()]
                        );
                        if (s == maxScoreShape) {
                            // change left multinote to merged multinote
                            // because the relnotes are sorted in same way as multinotes,
                            // we can prove that the first relnote in the new shape is correspond to the first relnote in left multinote's original shape
                            corpus.piecesMN[i][j][k].unit = 
                                shapeDict[corpus.piecesMN[i][j][k].getShapeIndex()][0].relDur * corpus.piecesMN[i][j][k].unit / maxScoreShape[0].relDur;
                            corpus.piecesMN[i][j][k].setShapeIndex(newShapeIndex);

                            // mark right multinote to be removed by have vel set to 0
                            corpus.piecesMN[i][j][k+n].vel = 0;
                            break;
                        }
                    }
                }
                // "delete" multinotes with vel == 0
                // to prevent that back() is also to be removed, we iterate from back to front
                for (int k = corpus.piecesMN[i][j].size() - 1; k >= 0; --k) {
                    if (corpus.piecesMN[i][j][k].vel == 0) {
                        corpus.piecesMN[i][j][k] = corpus.piecesMN[i][j].back();
                        corpus.piecesMN[i][j].pop_back();
                    }
                }
                sort(corpus.piecesMN[i][j].begin(), corpus.piecesMN[i][j].end());
            }
        }

        multinoteCount = 0;
        #pragma omp parallel for reduction(+:multinoteCount)
        for (int i = 0; i < corpus.piecesMN.size(); ++i) {
            for (int j = 0; j < corpus.piecesMN[i].size(); ++j) {
                multinoteCount += corpus.piecesMN[i][j].size();
            }
        }
        cout << ". Multinote count: " << multinoteCount << ", ";
        
        shapeCount.clear();
        // corpus.shrink();
        cout << "Time: " << (unsigned int) time(0) - iterTime;
        cout << endl;
    }

    // write vocab file
    writeShapeVocabFile(vocabFile, shapeDict);
    
    cout << "Write vocabs file done." << endl;

    writeOutputCorpusFile(outputCorpusFile, corpus, shapeDict, maxTrackNum, positionMethod);

    cout << "Write tokenized corpus file done. Total used time:" << time(0)-begTime << endl;
    return 0;
}
