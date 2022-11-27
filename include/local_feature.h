//
// Created by rcxxx on 22-11-24.
//

#ifndef CAT_RECOGNITION_FEATURE_H
#define CAT_RECOGNITION_FEATURE_H

#include <iostream>
#include <vector>

struct local_feature{
    std::string id;
    std::vector<float> data_row;
};

static std::vector<local_feature> loadFeature(std::ifstream& handle){
    std::vector<local_feature> output;
    std::string line;
    while (getline(handle, line)){
        local_feature f;
        std::istringstream read_str(line);
        std::string str;
        while (getline(read_str, str, ' ')){
            const char* number_f = str.data();
            try {
                f.data_row.push_back(std::stof(number_f));
            }
            catch (std::invalid_argument&){
                f.id = str;
                continue;
            }
        }
        output.emplace_back(f);
    }

    return output;
}

#endif //CAT_RECOGNITION_FEATURE_H
