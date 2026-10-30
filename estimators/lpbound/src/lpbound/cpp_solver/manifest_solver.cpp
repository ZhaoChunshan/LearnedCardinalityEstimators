#include <fstream>
#include <iostream>
#include "Highs.h"
#include <nlohmann/json.hpp>
using json = nlohmann::json;
#include <vector>
#include <chrono>
#include <unordered_map>
#include <cstring>
#include <map>
#include <iomanip>
#include <cstdint>
#include <functional>

#include <filesystem>

// Thread-local storage for solver instances and memory buffers
thread_local Highs* tl_highs = nullptr;
thread_local std::vector<int> tl_start;
thread_local std::vector<int> tl_index;
thread_local std::vector<double> tl_values;
// Pre-allocated LP vectors to avoid repeated allocations
thread_local std::vector<double> tl_col_cost;
thread_local std::vector<double> tl_col_lower;
thread_local std::vector<double> tl_col_upper;
thread_local std::vector<double> tl_row_lower;
thread_local std::vector<double> tl_row_upper;
thread_local bool tl_initialized = false;
int solver_threads = 1;

// Initialize thread-local solver and memory buffers
void initialize_thread_local_resources() {
    if (!tl_initialized) {
        tl_highs = new Highs();
        // Set solver options once per thread
        tl_highs->setOptionValue("presolve", "off");
        tl_highs->setOptionValue("output_flag", false);
        tl_highs->setOptionValue("log_to_console", false);
        tl_highs->setOptionValue("write_solution_to_file", false);
        tl_highs->setOptionValue("threads", solver_threads);
        
        // Pre-allocate memory buffers with reasonable capacity
        tl_start.reserve(1000);
        tl_index.reserve(10000);
        tl_values.reserve(10000);
        
        // Pre-allocate LP vectors
        tl_col_cost.reserve(1000);
        tl_col_lower.reserve(1000);
        tl_col_upper.reserve(1000);
        tl_row_lower.reserve(1000);
        tl_row_upper.reserve(1000);
        
        tl_initialized = true;
    }
}

// Cleanup function for thread-local resources
void cleanup_thread_local_resources() {
    if (tl_highs) {
        delete tl_highs;
        tl_highs = nullptr;
    }
    tl_initialized = false;
}

struct ManifestSubquery {
    struct LinearConstraint {
        std::unordered_map<int, double> variables;
        char sense;
        double rhs;
    };

    int subquery_id;
    int query_id;
    int subquery_ordinal;
    std::vector<LinearConstraint> constraints;
    std::unordered_map<int, std::string> variables;
    std::unordered_map<int, double> objective;
};

std::filesystem::path manifest_directory;

std::string input_path(const std::string& value) {
    const std::filesystem::path path(value);
    return (path.is_absolute() ? path : manifest_directory / path).lexically_normal().string();
}

json read_json_file(const std::string& path) {
    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("Could not open JSON file: " + path);
    }
    return json::parse(input);
}

ManifestSubquery parse_manifest_subquery(const json& row, bool inline_model) {
    ManifestSubquery subquery;
    subquery.subquery_id = row.at("subquery_id").get<int>();
    subquery.query_id = row.at("query_id").get<int>();
    subquery.subquery_ordinal = row.at("subquery_ordinal").get<int>();

    json constraints =
        inline_model ? row.at("constraints") : read_json_file(input_path(row.at("constraints").get<std::string>()));
    for (const auto& constraint : constraints) {
        ManifestSubquery::LinearConstraint parsed;
        for (const auto& [key, value] : constraint.at("variables").items()) {
            parsed.variables[std::stoi(key)] = value.get<double>();
        }
        const std::string sense = constraint.at("sense").get<std::string>();
        if (sense != "L" && sense != "G" && sense != "E") {
            throw std::runtime_error("Unsupported manifest constraint sense: " + sense);
        }
        parsed.sense = sense[0];
        parsed.rhs = constraint.at("rhs").get<double>();
        subquery.constraints.push_back(std::move(parsed));
    }

    json variables =
        inline_model ? row.at("variables") : read_json_file(input_path(row.at("variables").get<std::string>()));
    for (const auto& [key, value] : variables.items()) {
        subquery.variables[std::stoi(key)] = value.get<std::string>();
    }

    json objective =
        inline_model ? row.at("objective") : read_json_file(input_path(row.at("objective").get<std::string>()));
    for (const auto& [key, value] : objective.items()) {
        subquery.objective[std::stoi(key)] = value.get<double>();
    }
    return subquery;
}

std::vector<ManifestSubquery> load_manifest_subqueries(
    const std::string& manifest_path,
    std::string& benchmark) {
    json manifest = read_json_file(manifest_path);
    benchmark = manifest.at("benchmark").get<std::string>();
    std::vector<ManifestSubquery> result;
    result.reserve(manifest.at("subqueries").size());

    for (const auto& row : manifest.at("subqueries")) {
        result.push_back(parse_manifest_subquery(row, false));
    }
    return result;
}

bool for_each_jsonl_subquery(
    const std::string& jsonl_path,
    int expected_count,
    const std::function<bool(const ManifestSubquery&)>& callback) {
    std::ifstream input(jsonl_path);
    if (!input.is_open()) {
        throw std::runtime_error("Could not open packed manifest input: " + jsonl_path);
    }
    std::string line;
    int count = 0;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        ManifestSubquery subquery = parse_manifest_subquery(json::parse(line), true);
        ++count;
        if (!callback(subquery)) {
            return false;
        }
    }
    if (count != expected_count) {
        throw std::runtime_error(
            "Packed manifest count mismatch: expected " + std::to_string(expected_count) +
            ", got " + std::to_string(count));
    }
    return true;
}

std::vector<double> build_and_solve_manifest_lp(const ManifestSubquery& subquery) {
    const auto begin_build = std::chrono::steady_clock::now();
    initialize_thread_local_resources();

    const int num_vars = static_cast<int>(subquery.variables.size());
    const int num_rows = static_cast<int>(subquery.constraints.size());
    int num_nonzeros = 0;
    for (const auto& constraint : subquery.constraints) {
        num_nonzeros += static_cast<int>(constraint.variables.size());
    }

    tl_highs->clearModel();
    HighsLp lp;
    lp.num_col_ = num_vars;
    lp.num_row_ = num_rows;
    lp.offset_ = 0.0;
    lp.sense_ = ObjSense::kMaximize;

    tl_col_cost.assign(num_vars, 0.0);
    for (const auto& [var_id, coefficient] : subquery.objective) {
        if (var_id < 0 || var_id >= num_vars) {
            throw std::runtime_error("Manifest objective contains an invalid variable ID");
        }
        tl_col_cost[var_id] = coefficient;
    }
    lp.col_cost_ = tl_col_cost;

    tl_col_lower.assign(num_vars, 0.0);
    tl_col_upper.assign(num_vars, 1e30);
    tl_row_lower.assign(num_rows, -1e30);
    tl_row_upper.assign(num_rows, 1e30);

    tl_start.clear();
    tl_index.clear();
    tl_values.clear();
    tl_start.reserve(num_rows + 1);
    tl_index.reserve(num_nonzeros);
    tl_values.reserve(num_nonzeros);
    tl_start.push_back(0);

    for (int row_id = 0; row_id < num_rows; ++row_id) {
        const auto& constraint = subquery.constraints[row_id];
        if (constraint.sense == 'L') {
            tl_row_upper[row_id] = constraint.rhs;
        } else if (constraint.sense == 'G') {
            tl_row_lower[row_id] = constraint.rhs;
        } else {
            tl_row_lower[row_id] = constraint.rhs;
            tl_row_upper[row_id] = constraint.rhs;
        }
        for (const auto& [var_id, coefficient] : constraint.variables) {
            if (var_id < 0 || var_id >= num_vars) {
                throw std::runtime_error("Manifest constraint contains an invalid variable ID");
            }
            tl_index.push_back(var_id);
            tl_values.push_back(coefficient);
        }
        tl_start.push_back(static_cast<int>(tl_index.size()));
    }

    lp.col_lower_ = tl_col_lower;
    lp.col_upper_ = tl_col_upper;
    lp.row_lower_ = tl_row_lower;
    lp.row_upper_ = tl_row_upper;
    lp.a_matrix_.format_ = MatrixFormat::kRowwise;
    lp.a_matrix_.start_ = tl_start;
    lp.a_matrix_.index_ = tl_index;
    lp.a_matrix_.value_ = tl_values;

    const HighsStatus pass_status = tl_highs->passModel(lp);
    if (pass_status == HighsStatus::kError) {
        throw std::runtime_error("HiGHS rejected a manifest LP model");
    }
    const auto end_build = std::chrono::steady_clock::now();
    const int64_t build_time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end_build - begin_build).count();

    const auto begin_solve = std::chrono::steady_clock::now();
    const HighsStatus solve_status = tl_highs->run();
    const HighsInfo& info = tl_highs->getInfo();
    const auto end_solve = std::chrono::steady_clock::now();
    const int64_t solve_time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end_solve - begin_solve).count();
    const bool optimal =
        solve_status != HighsStatus::kError &&
        tl_highs->getModelStatus() == HighsModelStatus::kOptimal;

    return {
        info.objective_function_value,
        static_cast<double>(build_time),
        static_cast<double>(solve_time),
        optimal ? 1.0 : 0.0,
    };
}

int run_manifest_sequential(
    const std::string& manifest_path,
    const std::string& output_dir,
    int warmup_runs,
    int measured_runs) {
    manifest_directory = std::filesystem::absolute(manifest_path).parent_path();
    std::string benchmark;
    bool packed_jsonl = false;
    std::string jsonl_path;
    int expected_count = 0;
    std::vector<ManifestSubquery> subqueries;
    try {
        const json manifest = read_json_file(manifest_path);
        benchmark = manifest.at("benchmark").get<std::string>();
        packed_jsonl = manifest.value("format", "files") == "jsonl";
        if (packed_jsonl) {
            jsonl_path = input_path(manifest.at("subqueries_file").get<std::string>());
            expected_count = manifest.at("count").get<int>();
        } else {
            subqueries = load_manifest_subqueries(manifest_path, benchmark);
            expected_count = static_cast<int>(subqueries.size());
        }
    } catch (const std::exception& error) {
        std::cerr << "Failed to load manifest: " << error.what() << std::endl;
        return 1;
    }

    std::cout << "Manifest benchmark: " << benchmark << std::endl;
    std::cout << "Subqueries: " << expected_count
              << (packed_jsonl ? " (streamed JSONL)" : " (loaded files)") << std::endl;
    std::cout << "Warm-up runs: " << warmup_runs
              << ", measured runs: " << measured_runs
              << ", HiGHS threads: " << solver_threads << std::endl;

    initialize_thread_local_resources();
    auto solve_warmup = [&](const ManifestSubquery& subquery) {
            std::vector<double> result = build_and_solve_manifest_lp(subquery);
            if (result.size() < 4 || result[3] != 1.0) {
                std::cerr << "Non-optimal warm-up solve for SubqueryID="
                          << subquery.subquery_id << std::endl;
                return false;
            }
            return true;
    };
    try {
        for (int warmup = 0; warmup < warmup_runs; ++warmup) {
            bool ok = true;
            if (packed_jsonl) {
                ok = for_each_jsonl_subquery(jsonl_path, expected_count, solve_warmup);
            } else {
                for (const auto& subquery : subqueries) {
                    if (!solve_warmup(subquery)) {
                        ok = false;
                        break;
                    }
                }
            }
            if (!ok) {
                cleanup_thread_local_resources();
                return 1;
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "Failed during warm-up input streaming: " << error.what() << std::endl;
        cleanup_thread_local_resources();
        return 1;
    }

    const std::string output_path = output_dir + "/cpp_sequential_raw.csv";
    std::filesystem::create_directories(output_dir);
    const std::string temporary_path = output_path + ".tmp";
    std::ofstream output(temporary_path);
    if (!output.is_open()) {
        std::cerr << "Could not open output file: " << output_path << std::endl;
        cleanup_thread_local_resources();
        return 1;
    }
    output << "Benchmark,SubqueryID,QueryID,SubqueryOrdinal,RunID,"
              "BuildTimeNs,SolveTimeNs,TotalTimeNs,LPObjective,SolverStatus\n";
    output << std::setprecision(17);

    for (int run = 1; run <= measured_runs; ++run) {
        std::cout << "Measured run " << run << "/" << measured_runs << std::endl;
        auto solve_measured = [&](const ManifestSubquery& subquery) {
            const auto begin_total = std::chrono::steady_clock::now();
            std::vector<double> result = build_and_solve_manifest_lp(subquery);
            const auto end_total = std::chrono::steady_clock::now();
            const int64_t total_time =
                std::chrono::duration_cast<std::chrono::nanoseconds>(end_total - begin_total).count();
            const bool optimal = result.size() >= 4 && result[3] == 1.0;
            output << benchmark << ","
                   << subquery.subquery_id << ","
                   << subquery.query_id << ","
                   << subquery.subquery_ordinal << ","
                   << run << ","
                   << static_cast<int64_t>(result[1]) << ","
                   << static_cast<int64_t>(result[2]) << ","
                   << total_time << ","
                   << result[0] << ","
                   << (optimal ? "Optimal" : "NotOptimal") << "\n";
            if (!optimal) {
                std::cerr << "Non-optimal measured solve for SubqueryID="
                          << subquery.subquery_id << ", RunID=" << run << std::endl;
                return false;
            }
            return true;
        };
        bool ok = true;
        try {
            if (packed_jsonl) {
                ok = for_each_jsonl_subquery(jsonl_path, expected_count, solve_measured);
            } else {
                for (const auto& subquery : subqueries) {
                    if (!solve_measured(subquery)) {
                        ok = false;
                        break;
                    }
                }
            }
        } catch (const std::exception& error) {
            std::cerr << "Failed during measured input streaming: " << error.what() << std::endl;
            ok = false;
        }
        if (!ok) {
            cleanup_thread_local_resources();
            return 1;
        }
        output.flush();
    }
    cleanup_thread_local_resources();
    output.close();
    std::filesystem::rename(temporary_path, output_path);
    std::cout << "Wrote sequential timing results: " << output_path << std::endl;
    return 0;
}

int main(int argc, char* argv[]) {
    std::string manifest_path, output_dir;
    int warmup_runs = 1, measured_runs = 5;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string option(argv[i]);
            if (option == "--help" || option == "-h") {
                std::cout << "Usage: lpbound_parallel --sequential --manifest FILE --output-dir DIR "
                             "[--warmup-runs 1] [--runs 5] [--threads 1]\n";
                return 0;
            }
            if (option == "--sequential") continue;
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + option);
            const std::string value(argv[++i]);
            if (option == "--manifest") manifest_path = value;
            else if (option == "--output-dir") output_dir = value;
            else if (option == "--warmup-runs") warmup_runs = std::stoi(value);
            else if (option == "--runs") measured_runs = std::stoi(value);
            else if (option == "--threads") solver_threads = std::stoi(value);
            else throw std::runtime_error("Unknown option: " + option);
        }
        if (manifest_path.empty() || output_dir.empty() || warmup_runs < 0 || measured_runs < 1 || solver_threads != 1)
            throw std::runtime_error("manifest/output-dir required; warmup >= 0, runs > 0, threads must be 1");
        return run_manifest_sequential(manifest_path, output_dir, warmup_runs, measured_runs);
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        cleanup_thread_local_resources();
        return 1;
    }
}
