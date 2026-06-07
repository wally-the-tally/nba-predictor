#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <iostream>
#include <stdexcept>

struct TrainingSample {
    std::vector<double> inputs;
    double output;
};

// CSV splitter (handles quoted commas)
std::vector<std::string> splitCSV(const std::string& line) {
    std::vector<std::string> result;
    std::string cell;
    bool inQuotes = false;

    for (char c : line) {
        if (c == '"') {
            inQuotes = !inQuotes;
        }
        else if (c == ',' && !inQuotes) {
            result.push_back(cell);
            cell.clear();
        }
        else {
            cell += c;
        }
    }

    result.push_back(cell);
    return result;
}

std::vector<TrainingSample> generateTrainingData() {

    std::unordered_map<int, double> teams = {
        {1610612737,0.0},{1610612738,1.0},{1610612739,2.0},
        {1610612740,3.0},{1610612741,4.0},{1610612742,5.0},
        {1610612743,6.0},{1610612744,7.0},{1610612745,8.0},
        {1610612746,9.0},{1610612747,10.0},{1610612748,11.0},
        {1610612749,12.0},{1610612750,13.0},{1610612751,14.0},
        {1610612752,15.0},{1610612753,16.0},{1610612754,17.0},
        {1610612755,18.0},{1610612756,19.0},{1610612757,20.0},
        {1610612758,21.0},{1610612759,22.0},{1610612760,23.0},
        {1610612761,24.0},{1610612762,25.0},{1610612763,26.0},
        {1610612764,27.0},{1610612765,28.0},{1610612766,29.0}
    };

    std::vector<TrainingSample> data;

    std::ifstream file("Games.csv");
    if (!file.is_open()) {
        throw std::runtime_error("Could not open Games.csv");
    }

    std::string line;

    // Read header
    getline(file, line);
    auto header = splitCSV(line);

    std::unordered_map<std::string, int> colIndex;
    for (int i = 0; i < (int)header.size(); i++) {
        colIndex[header[i]] = i;
    }

    int debugCount = 0;

    while (getline(file, line)) {

        auto columns = splitCSV(line);

        if (columns.size() != header.size()) {
            continue;
        }

        try {
            int homeTeam = std::stoi(columns[colIndex["hometeamId"]]);
            int awayTeam = std::stoi(columns[colIndex["awayteamId"]]);
            int winner   = std::stoi(columns[colIndex["winner"]]);

            auto h = teams.find(homeTeam);
            auto a = teams.find(awayTeam);

            if (h == teams.end() || a == teams.end()) {
                continue;
            }

            TrainingSample sample;

            sample.inputs.push_back(h->second);
            sample.inputs.push_back(a->second);

            sample.output = (winner == homeTeam) ? 1.0 : 0.0;

            // 🔥 SANITY CHECK: print first 15 rows only
            if (debugCount < 15) {
                std::cout
                    << "Row " << debugCount << " | "
                    << "home=" << homeTeam << " "
                    << "away=" << awayTeam << " "
                    << "winner=" << winner << " | "
                    << "input=(" << h->second << "," << a->second << ") "
                    << "output=" << sample.output
                    << "\n";

                debugCount++;
            }

            data.push_back(sample);

        } catch (...) {
            continue;
        }
    }

    return data;
}