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

int main(int argc, char *argv[]) {
    // read and validate args
    if (argc != 6) {
        cout << "Must have 5 arguments: [inCorpusDirPath] [outCorpusDirPath] [bpeIter] [countingMethod] [samplingRate]" << endl;
        return 1;
    }
    string inCorpusDirPath(argv[1]);
    string outCorpusDirPath(argv[2]);
    int bpeIter = atoi(argv[3]);
    string countingMethod(argv[4]);
    double samplingRate = atof(argv[5]);
    if (bpeIter <= 0 || 2045 < bpeIter) {
        cout << "Error: bpeIter <= 0 or > 2045: " << bpeIter << endl;
        return 1;
    }
    if (countingMethod != "ours" && countingMethod != "symphonynet" && countingMethod != "wordpiece") {
        cout << "Error: countingMethod is not ( ours | symphonynet | wordpiece ): " << countingMethod << endl;
        return 1;
    }
    cout << "inCorpusDirPath: " << inCorpusDirPath << endl
        << "outCorpusDirPath: " << outCorpusDirPath << endl
        << "bpeIter: " << bpeIter << endl
        << "countingMethod: " << countingMethod << endl
        << "samplingRate: " << samplingRate << endl;

    // open files
    string inCorpusFilePath = inCorpusDirPath + "/corpus";
    cout << "Input corpus file path: " << inCorpusFilePath << endl;
    ifstream inCorpusFile(inCorpusFilePath, ios::in | ios::binary);
    if (!inCorpusFile.is_open()) {
        cout << "Failed to open corpus file: " << inCorpusFilePath << endl;
        return 1;
    }
    
    string parasFilePath = inCorpusDirPath + "/paras";
    cout << "Input parameter file path: " << parasFilePath << endl;
    ifstream parasFile(parasFilePath, ios::in | ios::binary);
    if (!parasFile.is_open()) {
        cout << "Failed to open parameters file: " << parasFilePath << endl;
        return 1;
    }

    string vocabFilePath = outCorpusDirPath + "/shape_vocab";
    cout << "Output shape vocab file path: " << vocabFilePath << endl;
    ofstream vocabFile(vocabFilePath, ios::out | ios::trunc);
    if (!vocabFile.is_open()) {
        cout << "Failed to open vocab output file: " << vocabFilePath << endl;
        return 1;
    }

    string outCorpusFilePath = outCorpusDirPath + "/corpus";
    cout << "Output merged corpus file path: " << outCorpusFilePath << endl;
    ofstream outCorpusFile(outCorpusFilePath, ios::out | ios::trunc);
    if (!outCorpusFile.is_open()) {
        cout << "Failed to open tokenized corpus output file: " << outCorpusFilePath << endl;
        return 1;
    }

    time_t begTime = time(0);
    cout << "Reading input files" << endl;

    // read parameters
    map<string, string> paras = readParasFile(parasFile);
    int nth, maxDur, maxTrackNum;
    string positionMethod;
    // stoi, c++11 thing
    nth = stoi(paras[string("nth")]);
    // cout << "nth: " << nth << endl;
    maxDur = stoi(paras[string("max_duration")]);
    // cout << "maxDur: " << maxDur << endl;
    maxTrackNum = stoi(paras[string("max_track_number")]);
    // cout << "maxTrackNum: " << maxTrackNum << endl;
    positionMethod = paras[string("position_method")];
    // cout << "positionMethod=" << positionMethod << endl;
    if (nth <= 0 || maxDur <= 0 || maxDur > 255 || maxTrackNum <= 0 || (positionMethod != "event" && positionMethod != "attribute")) {
        cout << "Corpus parameter error" << endl;
        return 1;
    }

    // read notes from corpus
    Corpus corpus = readCorpusFile(inCorpusFile, nth, positionMethod);
    vector<Shape> shapeDict;
    shapeDict.reserve(bpeIter + 2);
    shapeDict.push_back({RelNote(0, 0, 0, 1)}); // DEFAULT_SHAPE_END
    shapeDict.push_back({RelNote(1, 0, 0, 1)}); // DEFAULT_SHAPE_CONT

    // sort and count notes
    size_t multinoteCount = 0;
    size_t drumMultinoteCount = 0;
    for (int i = 0; i < corpus.piecesMN.size(); ++i) {
        for (int j = 0; j < corpus.piecesMN[i].size(); ++j) {
            if (corpus.piecesTN[i][j] == 128) {
                drumMultinoteCount += corpus.piecesMN[i][j].size();
            }
            multinoteCount += corpus.piecesMN[i][j].size();
            sort(corpus.piecesMN[i][j].begin(), corpus.piecesMN[i][j].end());
        }
    }
    double avgMulpi = calculateAvgMulpiSize(corpus);
    // printTrack(corpus[0][0], shapeDict, 0, 10);

    cout << "Multinote count: " << multinoteCount
        << ", Drum's multinote count: " << drumMultinoteCount
        << ", Average mulpi: " << avgMulpi
        << ", Time: " << (unsigned int) time(0) - begTime << endl;
    if (multinoteCount == 0 || multinoteCount == drumMultinoteCount) {
        cout << "No notes to merge. Exited." << endl;
        return 1;
    }

    time_t iterTime;
    map<Shape, unsigned int> shapeScore;
    for (int iterCount = 0; iterCount < bpeIter; ++iterCount) {
        cout << "Iter:" << iterCount << ", ";
        iterTime = time(0);

        updateNeighbor(corpus, shapeDict);

        // count shape frequency
        if (countingMethod == "ours") {
            oursShapeCounting(corpus, shapeDict, shapeScore, samplingRate);
        }
        else if (countingMethod == "symphonynet") {
            symphonyNetShapeCounting(corpus, shapeDict, shapeScore, samplingRate);
        }
        else {
            wordPieceScoreShapeCounting(corpus, shapeDict, shapeScore, samplingRate);
        }
        cout << "Find " << shapeScore.size() << " unique pairs" << ", ";
        if (shapeScore.size() == 0) {
            cout << "Early termination: no shapes found" << endl;
            break;
        }
    
        // add shape with highest frequency into shapeDict
        Shape maxScoreShape;
        int maxScore = 0;
        for (auto it = shapeScore.cbegin(); it != shapeScore.cend(); it++) {
            if (maxScore < (*it).second) {
                maxScoreShape = (*it).first;
                maxScore = (*it).second;
            }
            // cout << shape2str((*it).first) << endl;
        }

        unsigned int newShapeIndex = shapeDict.size();
        shapeDict.push_back(maxScoreShape);
        cout << "New shape: " << shape2str(maxScoreShape) << " Score=" << maxScore;

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
                        // if (shapeDict[corpus.piecesMN[i][j][k].getShapeIndex()].size()
                        //         + shapeDict[corpus.piecesMN[i][j][k+n].getShapeIndex()].size()
                        //         != maxScoreShape.size()) {
                        //     continue;
                        // }
                        Shape s = getShapeOfMultiNotePair(
                            corpus.piecesMN[i][j][k],
                            corpus.piecesMN[i][j][k+n],
                            shapeDict[corpus.piecesMN[i][j][k].getShapeIndex()],
                            shapeDict[corpus.piecesMN[i][j][k+n].getShapeIndex()]
                        );
                        if (s == maxScoreShape) {
                            // change left multinote to merged multinote
                            // because the relnotes are sorted in same way as multinotes,
                            // the first relnote in the new shape is correspond to the first relnote in left multinote's original shape
                            uint8_t newUnit = shapeDict[corpus.piecesMN[i][j][k].getShapeIndex()][0].relDur * corpus.piecesMN[i][j][k].unit / maxScoreShape[0].relDur;
                            // unit cannot be greater than max_duration
                            if (newUnit > maxDur) break;
                            corpus.piecesMN[i][j][k].unit = newUnit;
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
        
        shapeScore.clear();
        // corpus.shrink();
        cout << "Time: " << (unsigned int) time(0) - iterTime << endl;
    }

    avgMulpi = calculateAvgMulpiSize(corpus);
    cout << "Average mulpi: " << avgMulpi << endl;

    // write vocab file
    writeShapeVocabFile(vocabFile, shapeDict);
    
    cout << "Write vocabs file done." << endl;

    writeOutputCorpusFile(outCorpusFile, corpus, shapeDict, maxTrackNum, positionMethod);

    cout << "Write tokenized corpus file done. Total used time: " << time(0)-begTime << endl;
    return 0;
}
