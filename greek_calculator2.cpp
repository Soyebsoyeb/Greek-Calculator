#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <thread>
#include <future>
#include <mutex>
#include <sstream>
#include <iomanip>
#include <map>
#include <numeric>
#include <chrono>
#include <atomic>
#include <queue>
#include <condition_variable>

using namespace std;
using namespace chrono;

// Advanced configuration for batch processing
struct BatchConfig {
    int threads = thread::hardware_concurrency();
    bool enable_advanced_greeks = true;
    bool enable_risk_metrics = true;
    bool enable_portfolio_analysis = true;
    size_t batch_size = 1000;
    double risk_free_rate = 0.05;
};

// Advanced math functions with better precision
class MathUtils {
public:
    static double norm_pdf(double x) {
        return 0.3989422804014327 * exp(-0.5 * x * x);
    }
    
    static double norm_cdf(double x) {
        if (x < -8.0) return 0.0;
        if (x > 8.0) return 1.0;
        
        double sum = x;
        double term = x;
        for (int i = 1; i < 100; i++) {
            term *= x * x / (2 * i + 1);
            sum += term;
        }
        return 0.5 + (sum * 0.3989422804014327);
    }
    
    static double calculate_implied_volatility(double market_price, double S, double K, 
                                              double r, double T, bool isCall, 
                                              double initial_guess = 0.2, int max_iter = 100) {
        double sigma = initial_guess;
        for (int i = 0; i < max_iter; i++) {
            double price = black_scholes_price(S, K, r, sigma, T, isCall);
            double diff = price - market_price;
            if (abs(diff) < 1e-8) return sigma;
            
            // Simple bisection method
            if (diff > 0) sigma *= 0.9;
            else sigma *= 1.1;
            
            if (sigma > 5.0) sigma = 5.0;
            if (sigma < 0.001) sigma = 0.001;
        }
        return sigma;
    }
    
    static double black_scholes_price(double S, double K, double r, double sigma, double T, bool isCall) {
        if (T <= 0.0) {
            return isCall ? max(S - K, 0.0) : max(K - S, 0.0);
        }
        
        double sqrtT = sqrt(T);
        double d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
        double d2 = d1 - sigma * sqrtT;
        
        if (isCall) {
            return S * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2);
        } else {
            return K * exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1);
        }
    }
};

// Advanced Greeks with second-order metrics
struct AdvancedGreeks {
    // First order
    double delta;
    double theta;  // per day
    double vega;   // per 1%
    double rho;    // per 1%
    
    // Second order
    double gamma;
    double vanna;   // dDelta/dVol
    double charm;   // dDelta/dTime
    double vomma;   // dVega/dVol
    double veta;    // dVega/dTime
    double speed;   // dGamma/dSpot
    double zomma;   // dGamma/dVol
    
    // Risk metrics
    double dollar_gamma;
    double dollar_delta;
};

// Portfolio risk metrics
struct PortfolioRisk {
    double total_delta;
    double total_gamma;
    double total_vega;
    double total_theta;
    double total_rho;
    
    double delta_exposure;
    double gamma_exposure;
    double vega_exposure;
    
    double var_95;  // Value at Risk 95%
    double expected_shortfall;
    
    map<string, double> greek_contributions;
};

// Enhanced input data structure
struct OptionData {
    string id;
    string underlying;
    double spot;
    double strike;
    double rate;
    double volatility;
    double time;
    double dividend_yield;
    bool isCall;
    double market_price;  // For implied vol calculation
    int quantity;
    double notional;
    
    // Constructor for easy initialization
    OptionData(string id_, double s, double k, double r, double vol, double t, bool call)
        : id(id_), spot(s), strike(k), rate(r), volatility(vol), time(t), isCall(call),
          dividend_yield(0.0), market_price(0.0), quantity(1), notional(s) {}
};

// Comprehensive output structure
struct OptionResult {
    string id;
    double theoretical_price;
    double market_price;
    AdvancedGreeks greeks;
    double intrinsic_value;
    double extrinsic_value;
    double time_value;
    double implied_volatility;
    double price_diff;  // Theoretical vs market
    
    // Risk metrics
    double delta_exposure;
    double gamma_exposure;
    double vega_exposure;
    
    bool has_market_data;
    string status;
};

// Thread-safe logger
class Logger {
private:
    mutex log_mutex;
public:
    void log(const string& message) {
        lock_guard<mutex> lock(log_mutex);
        auto now = system_clock::now();
        auto time_t = system_clock::to_time_t(now);
        cout << "[" << put_time(localtime(&time_t), "%H:%M:%S") << "] " << message << endl;
    }
    
    void progress(size_t current, size_t total) {
        lock_guard<mutex> lock(log_mutex);
        double percentage = (static_cast<double>(current) / total) * 100.0;
        cout << "\rProgress: " << current << "/" << total << " (" << fixed << setprecision(1) << percentage << "%)";
        cout.flush();
        if (current == total) cout << endl;
    }
};

// Advanced Greeks calculator
class GreekCalculator {
public:
    static AdvancedGreeks calculate_advanced_greeks(const OptionData& data) {
        AdvancedGreeks greeks;
        
        if (data.time <= 0.0) {
            // Handle expiration case
            greeks.delta = data.isCall ? (data.spot > data.strike ? 1.0 : 0.0) 
                                      : (data.spot < data.strike ? -1.0 : 0.0);
            greeks.gamma = greeks.vega = greeks.theta = greeks.rho = 0.0;
            greeks.vanna = greeks.charm = greeks.vomma = greeks.veta = greeks.speed = greeks.zomma = 0.0;
            return greeks;
        }
        
        double S = data.spot;
        double K = data.strike;
        double r = data.rate;
        double sigma = data.volatility;
        double T = data.time;
        double q = data.dividend_yield;
        
        double sqrtT = sqrt(T);
        double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
        double d2 = d1 - sigma * sqrtT;
        double pdf_d1 = MathUtils::norm_pdf(d1);
        
        // First order Greeks
        greeks.delta = data.isCall ? exp(-q * T) * MathUtils::norm_cdf(d1) 
                                  : exp(-q * T) * (MathUtils::norm_cdf(d1) - 1.0);
        
        greeks.gamma = exp(-q * T) * pdf_d1 / (S * sigma * sqrtT);
        greeks.vega = S * exp(-q * T) * pdf_d1 * sqrtT * 0.01; // per 1%
        
        // Theta (per day)
        double term1 = -(S * exp(-q * T) * pdf_d1 * sigma) / (2.0 * sqrtT);
        if (data.isCall) {
            greeks.theta = (term1 + q * S * exp(-q * T) * MathUtils::norm_cdf(d1) 
                          - r * K * exp(-r * T) * MathUtils::norm_cdf(d2)) / 365.0;
        } else {
            greeks.theta = (term1 - q * S * exp(-q * T) * MathUtils::norm_cdf(-d1) 
                          + r * K * exp(-r * T) * MathUtils::norm_cdf(-d2)) / 365.0;
        }
        
        // Rho (per 1%)
        greeks.rho = data.isCall ? (K * T * exp(-r * T) * MathUtils::norm_cdf(d2)) * 0.01 
                                : (-K * T * exp(-r * T) * MathUtils::norm_cdf(-d2)) * 0.01;
        
        // Second order Greeks
        greeks.vanna = -exp(-q * T) * pdf_d1 * d2 / sigma;
        greeks.charm = -exp(-q * T) * pdf_d1 * (2 * (r - q) * T - d2 * sigma * sqrtT) / (2 * T * sigma * sqrtT);
        greeks.vomma = S * exp(-q * T) * pdf_d1 * sqrtT * d1 * d2 / sigma;
        greeks.speed = -greeks.gamma / S * (1 + d1 / (sigma * sqrtT));
        greeks.zomma = greeks.gamma * (d1 * d2 - 1) / sigma;
        
        // Risk exposures
        greeks.dollar_delta = greeks.delta * S;
        greeks.dollar_gamma = 0.5 * greeks.gamma * S * S;
        
        return greeks;
    }
};

// Portfolio risk analyzer
class RiskAnalyzer {
public:
    static PortfolioRisk analyze_portfolio(const vector<OptionResult>& results, 
                                          const vector<OptionData>& data) {
        PortfolioRisk risk;
        
        // Aggregate Greeks
        risk.total_delta = risk.total_gamma = risk.total_vega = risk.total_theta = risk.total_rho = 0.0;
        
        for (size_t i = 0; i < results.size(); i++) {
            int quantity = data[i].quantity;
            risk.total_delta += results[i].greeks.delta * quantity;
            risk.total_gamma += results[i].greeks.gamma * quantity;
            risk.total_vega += results[i].greeks.vega * quantity;
            risk.total_theta += results[i].greeks.theta * quantity;
            risk.total_rho += results[i].greeks.rho * quantity;
        }
        
        // Calculate exposures
        risk.delta_exposure = risk.total_delta * data[0].spot;  // Approximate
        risk.gamma_exposure = 0.5 * risk.total_gamma * data[0].spot * data[0].spot;
        risk.vega_exposure = risk.total_vega * 100;  // For 1% vol change
        
        // Simple VaR calculation (95% confidence)
        vector<double> pnl_distribution;
        for (const auto& result : results) {
            // Simulate price changes
            double pnl_change = result.greeks.delta * data[0].spot * 0.01; // 1% spot move
            pnl_distribution.push_back(pnl_change);
        }
        
        sort(pnl_distribution.begin(), pnl_distribution.end());
        size_t var_index = static_cast<size_t>(pnl_distribution.size() * 0.05);
        risk.var_95 = pnl_distribution[var_index];
        
        // Expected shortfall (average of worst 5%)
        double es_sum = 0.0;
        for (size_t i = 0; i < var_index; i++) {
            es_sum += pnl_distribution[i];
        }
        risk.expected_shortfall = es_sum / var_index;
        
        return risk;
    }
};

// Thread-safe batch processor
class BatchProcessor {
private:
    BatchConfig config;
    Logger logger;
    atomic<size_t> processed_count{0};
    
public:
    BatchProcessor(const BatchConfig& cfg) : config(cfg) {}
    
    vector<OptionResult> process_batch_parallel(const vector<OptionData>& batch) {
        vector<OptionResult> results(batch.size());
        vector<thread> threads;
        size_t chunk_size = (batch.size() + config.threads - 1) / config.threads;
        
        auto process_chunk = [&](size_t start, size_t end, int thread_id) {
            for (size_t i = start; i < end && i < batch.size(); ++i) {
                try {
                    results[i] = process_single_option(batch[i]);
                    size_t count = processed_count.fetch_add(1) + 1;
                    if (count % 100 == 0) {
                        logger.progress(count, batch.size());
                    }
                } catch (const exception& e) {
                    logger.log("Thread " + to_string(thread_id) + " error: " + e.what());
                }
            }
        };
        
        logger.log("Starting parallel processing with " + to_string(config.threads) + " threads");
        auto start_time = high_resolution_clock::now();
        
        for (int i = 0; i < config.threads; ++i) {
            size_t start = i * chunk_size;
            size_t end = min((i + 1) * chunk_size, batch.size());
            threads.emplace_back(process_chunk, start, end, i);
        }
        
        for (auto& t : threads) {
            t.join();
        }
        
        auto end_time = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end_time - start_time);
        
        logger.log("Processing completed in " + to_string(duration.count()) + " ms");
        return results;
    }
    
private:
    OptionResult process_single_option(const OptionData& data) {
        OptionResult result;
        result.id = data.id;
        result.market_price = data.market_price;
        result.has_market_data = (data.market_price > 0);
        
        try {
            // Calculate theoretical price
            result.theoretical_price = MathUtils::black_scholes_price(
                data.spot, data.strike, data.rate, data.volatility, data.time, data.isCall);
            
            // Calculate advanced Greeks
            result.greeks = GreekCalculator::calculate_advanced_greeks(data);
            
            // Calculate intrinsic/extrinsic values
            if (data.isCall) {
                result.intrinsic_value = max(data.spot - data.strike, 0.0);
            } else {
                result.intrinsic_value = max(data.strike - data.spot, 0.0);
            }
            result.extrinsic_value = result.theoretical_price - result.intrinsic_value;
            result.time_value = result.extrinsic_value;
            
            // Calculate implied volatility if market price available
            if (result.has_market_data) {
                result.implied_volatility = MathUtils::calculate_implied_volatility(
                    data.market_price, data.spot, data.strike, data.rate, data.time, data.isCall);
                result.price_diff = result.theoretical_price - data.market_price;
            }
            
            // Calculate risk exposures
            result.delta_exposure = result.greeks.delta * data.spot * data.quantity;
            result.gamma_exposure = 0.5 * result.greeks.gamma * data.spot * data.spot * data.quantity;
            result.vega_exposure = result.greeks.vega * data.quantity;
            
            result.status = "SUCCESS";
            
        } catch (const exception& e) {
            result.status = "ERROR: " + string(e.what());
        }
        
        return result;
    }
};

// CSV parser with enhanced error handling
class CSVParser {
public:
    static vector<OptionData> parse_input_file(const string& filename) {
        ifstream file(filename);
        if (!file.is_open()) {
            throw runtime_error("Cannot open file: " + filename);
        }
        
        vector<OptionData> options;
        string line;
        int line_num = 0;
        
        // Read header
        getline(file, line);
        auto headers = split_csv_line(line);
        
        while (getline(file, line)) {
            line_num++;
            if (line.empty()) continue;
            
            try {
                auto fields = split_csv_line(line);
                OptionData data = parse_option_row(fields, headers);
                options.push_back(data);
            } catch (const exception& e) {
                cerr << "Warning: Skipping line " << line_num << " - " << e.what() << endl;
            }
        }
        
        return options;
    }
    
private:
    static vector<string> split_csv_line(const string& line) {
        vector<string> result;
        stringstream ss(line);
        string item;
        
        while (getline(ss, item, ',')) {
            // Remove quotes and trim whitespace
            if (!item.empty()) {
                if (item.front() == '"' && item.back() == '"') {
                    item = item.substr(1, item.size() - 2);
                }
                // Trim whitespace
                item.erase(0, item.find_first_not_of(" \t"));
                item.erase(item.find_last_not_of(" \t") + 1);
            }
            result.push_back(item);
        }
        
        return result;
    }
    
    static OptionData parse_option_row(const vector<string>& fields, const vector<string>& headers) {
        map<string, string> field_map;
        for (size_t i = 0; i < min(headers.size(), fields.size()); ++i) {
            field_map[to_lower(headers[i])] = fields[i];
        }
        
        OptionData data(
            get_field(field_map, "id", "OPT_" + to_string(rand() % 10000)),
            stod(get_field(field_map, "spot")),
            stod(get_field(field_map, "strike")),
            stod(get_field(field_map, "rate", "0.05")),
            stod(get_field(field_map, "volatility")),
            stod(get_field(field_map, "time")),
            get_field(field_map, "type", "call") == "call"
        );
        
        data.underlying = get_field(field_map, "underlying", "STOCK");
        data.dividend_yield = stod(get_field(field_map, "dividend_yield", "0.0"));
        data.market_price = stod(get_field(field_map, "market_price", "0.0"));
        data.quantity = stoi(get_field(field_map, "quantity", "1"));
        data.notional = stod(get_field(field_map, "notional", to_string(data.spot)));
        
        return data;
    }
    
    static string get_field(const map<string, string>& field_map, const string& key, const string& default_val = "") {
        auto it = field_map.find(to_lower(key));
        return it != field_map.end() ? it->second : default_val;
    }
    
    static string to_lower(const string& s) {
        string result = s;
        transform(result.begin(), result.end(), result.begin(), ::tolower);
        return result;
    }
};

// Enhanced results writer
class ResultsWriter {
public:
    static void write_comprehensive_results(const vector<OptionResult>& results, 
                                          const string& output_file) {
        ofstream file(output_file);
        if (!file.is_open()) {
            throw runtime_error("Cannot open output file: " + output_file);
        }
        
        // Write comprehensive header
        file << "ID,TheoreticalPrice,MarketPrice,PriceDifference,ImpliedVol,"
             << "Delta,Gamma,Vega,Theta,Rho,"
             << "Vanna,Charm,Vomma,Speed,Zomma,"
             << "IntrinsicValue,ExtrinsicValue,TimeValue,"
             << "DeltaExposure,GammaExposure,VegaExposure,"
             << "DollarDelta,DollarGamma,Status" << endl;
        
        file << fixed << setprecision(6);
        for (const auto& result : results) {
            file << result.id << ","
                 << result.theoretical_price << ","
                 << result.market_price << ","
                 << result.price_diff << ","
                 << result.implied_volatility << ","
                 << result.greeks.delta << ","
                 << result.greeks.gamma << ","
                 << result.greeks.vega << ","
                 << result.greeks.theta << ","
                 << result.greeks.rho << ","
                 << result.greeks.vanna << ","
                 << result.greeks.charm << ","
                 << result.greeks.vomma << ","
                 << result.greeks.speed << ","
                 << result.greeks.zomma << ","
                 << result.intrinsic_value << ","
                 << result.extrinsic_value << ","
                 << result.time_value << ","
                 << result.delta_exposure << ","
                 << result.gamma_exposure << ","
                 << result.vega_exposure << ","
                 << result.greeks.dollar_delta << ","
                 << result.greeks.dollar_gamma << ","
                 << "\"" << result.status << "\"" << endl;
        }
        
        file.close();
    }
    
    static void write_portfolio_report(const PortfolioRisk& risk, 
                                     const vector<OptionResult>& results,
                                     const string& report_file) {
        ofstream file(report_file);
        
        file << "PORTFOLIO RISK REPORT" << endl;
        file << "====================" << endl;
        file << "Total Options: " << results.size() << endl;
        file << fixed << setprecision(4);
        
        file << "\nGREEKS EXPOSURE:" << endl;
        file << "Delta: " << risk.total_delta << " (Exposure: $" << risk.delta_exposure << ")" << endl;
        file << "Gamma: " << risk.total_gamma << " (Exposure: $" << risk.gamma_exposure << ")" << endl;
        file << "Vega: " << risk.total_vega << " (Exposure: $" << risk.vega_exposure << " per 1%)" << endl;
        file << "Theta: " << risk.total_theta << " per day" << endl;
        file << "Rho: " << risk.total_rho << " per 1%" << endl;
        
        file << "\nRISK METRICS:" << endl;
        file << "VaR 95%: $" << risk.var_95 << endl;
        file << "Expected Shortfall: $" << risk.expected_shortfall << endl;
        
        file.close();
    }
};

// Main application
int main(int argc, char* argv[]) {
    if (argc < 3) {
        cout << "Advanced Options Batch Processor" << endl;
        cout << "Usage: " << argv[0] << " <input_csv> <output_csv> [threads]" << endl;
        cout << "Input CSV columns: ID,Spot,Strike,Rate,Volatility,Time,Type" << endl;
        cout << "Optional columns: Underlying,DividendYield,MarketPrice,Quantity,Notional" << endl;
        return 1;
    }
    
    string input_file = argv[1];
    string output_file = argv[2];
    
    BatchConfig config;
    if (argc >= 4) {
        config.threads = max(1, stoi(argv[3]));
    }
    
    try {
        // Parse input data
        auto options_data = CSVParser::parse_input_file(input_file);
        cout << "Loaded " << options_data.size() << " options" << endl;
        
        // Process batch
        BatchProcessor processor(config);
        auto results = processor.process_batch_parallel(options_data);
        
        // Write results
        ResultsWriter::write_comprehensive_results(results, output_file);
        
        // Generate portfolio report
        if (config.enable_portfolio_analysis && !results.empty()) {
            auto risk = RiskAnalyzer::analyze_portfolio(results, options_data);
            string report_file = output_file + ".risk_report.txt";
            ResultsWriter::write_portfolio_report(risk, results, report_file);
            cout << "Risk report: " << report_file << endl;
        }
        
        cout << "Results written to: " << output_file << endl;
        cout << "Successfully processed " << results.size() << " options" << endl;
        
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}
