//
//  LightGBM.hpp
//  ml_codes
//
//  Created by 柴田 智喜 on 2018/02/07.
//  Copyright © 2018年 柴田 智喜. All rights reserved.
//

#ifndef LightGBM_hpp
#define LightGBM_hpp

#include "clf_tree.hpp"
#include <random>
#include <tuple>
#include <cstdlib>
#include <cmath>
#include <sstream>

class LightGBM_clf{
///////////////////////////////
// クラス内クラスの定義
///////////////////////////////
public:
    // LightGBM_clfのパラメータ
    struct LightGBM_clf_param{
        CLF_tree::CLF_tree_param tree_param;
        int estimators_num;
        int random_seed;
        double high_gradient_sampling_ratio;
        double low_gradient_sampling_ratio;
        double max_gamma;
        double min_gamma;
        double gamma_precision;
        LightGBM_clf_param();
    };
    
    //Exclusive Feature Bundling
    class EFB{
    private:
        std::vector<std::vector<int>>bundles;
        std::vector<std::map<int, std::map<double, int>>>  bundles_map;
        
    protected:
        std::vector<int> non_0_count(const Eigen::MatrixXd& X);
        int conflict_count(const Eigen::MatrixXd& col_1,const Eigen::MatrixXd& col_2);
        std::vector<std::vector<int>> greedy_bundling(const Eigen::MatrixXd& X,int max_conflict_count);
        std::map<int, std::map<double,int>> bundle_map_builder(const Eigen::MatrixXd& X, std::vector<int>&bundle);
        Eigen::MatrixXd mapping(const Eigen::MatrixXd& X, std::vector<int>&bundle, std::map<int, std::map<double,int>>);
        
        
    public:
        //Eigen::MatrixXd static exclusive_feature_boundling(const Eigen::MatrixXd& X,int max_conflict = 0);
        void fit(const Eigen::MatrixXd& X,int max_conflict_count);
        Eigen::MatrixXd transform(const Eigen::MatrixXd& X);
        Eigen::MatrixXd fit_transform(const Eigen::MatrixXd& X,int max_conflict_count = 0);
        
        
        //テストクラスのフレンド宣言
        friend class LightGBMTest;
        FRIEND_TEST(LightGBMTest, non_0_count);
        FRIEND_TEST(LightGBMTest, conflict_count);
        FRIEND_TEST(LightGBMTest, greedy_bundling);
        FRIEND_TEST(LightGBMTest, bundle_map_builder);
        FRIEND_TEST(LightGBMTest, mapping);
        FRIEND_TEST(LightGBMTest, EFB_fit);
        FRIEND_TEST(LightGBMTest, transfrom);
        FRIEND_TEST(LightGBMTest, fit_transfrom);
        
    };
    
private:
    // 決定木の変更部分
    class LightGBM_CLF_tree:public CLF_tree{
    private:
        // 変数
        LightGBM_clf* forest;
        
        // メソッド
        double neg_variance_gain(std::vector<int>& more_idx,std::vector<int>& less_idx);
        void make_leaf_value(std::vector<int>& data_idx, Node* split_node);
        double leaf_solver(std::vector<int>& data_idx, double tree_label);
        double loss_func_for_leaf(std::vector<int>& data_idx, double tree_label, double gamma);
    
    protected:
        double evaluation(std::vector<int>& more_idx, std::vector<int>& less_idx);
        
    public:
        // 変数
        int forest_generation;
        double this_tree_label;
        
        // メソッド
        LightGBM_CLF_tree();
        ~LightGBM_CLF_tree();
        void set_forest(LightGBM_clf* forest_address);
        void set_leaf_value_on_root(double reaf_value);
        void fit(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y, std::vector<int>& data_idx);
        
        //テスト用
        Tree& get_tree(){return tree;};
        
        
        //テストクラスのフレンド宣言
        friend class LightGBMTest;
        FRIEND_TEST(LightGBMTest, set_leaf_value_on_root);
        FRIEND_TEST(LightGBMTest, init_0_generation_trees);
        FRIEND_TEST(LightGBMTest, make_cash_result);
        FRIEND_TEST(LightGBMTest, neg_variance_gain);
        FRIEND_TEST(LightGBMTest, loss_func_for_leaf);
        FRIEND_TEST(LightGBMTest, leaf_solver);
        FRIEND_TEST(LightGBMTest, fit);
        
    
    };
    
    //　乱数系
    struct Rand{
        std::mt19937 mt_rand;
        virtual std::map<double,std::vector<int>> GOS_sampling(std::map<double,std::vector<double>>& neg_gradient, double high_gradient_sampling_ratio, double low_gradient_sampling_ratio);
        const std::vector<int> random_sampling_without_replacement(int sample_num, int sampled_nums, int return_base = 0);
    };

/////////////////////////////
// メンバの定義
/////////////////////////////
protected:
    // 変数
    std::map<double,std::vector<LightGBM_CLF_tree>> forest;//forest[ラベル][世代].tree.getNode()->leaf_value = f_ラベル(x)を葉に持つ。
    LightGBM_clf_param param;
    const Eigen::MatrixXd* X_train;
    const Eigen::MatrixXd* y_train;
    std::vector<double> label_list;
    std::map<double,int> label_count;
    int data_count;
    std::map<double,Eigen::MatrixXd> cash_result;
    std::map<double,std::vector<double>> neg_gradient;
    std::map<double,std::vector<double>> neg_factored_gradient;
    
    //　メソッド
    void init_0_generation_trees(std::vector<double>& label_list, std::map<double,int>& label_count);
    std::map<double,Eigen::MatrixXd> make_cash_result(const Eigen::MatrixXd& X, int generation);//ラベルごとの木で、データそれぞれの出力を出す。
    std::map<double,std::vector<double>> neg_gradient_cross_entropy(const Eigen::MatrixXd& y);
    std::map<double,std::vector<double>> factor_neg_gradient(double high_gradient_sampling_ratio, double low_gradient_sampling_ratio);
    std::tuple<std::vector<double>, std::map<double, int>> count_label(const Eigen::MatrixXd& y_train);
    
    // 乱数系インスタンス
    Rand* rand;
    Rand rand_instance;
    
public:
    // 変数
    std::map<double, Eigen::MatrixXd>  current_probs_map;
    EFB exclusive_feature_bundling;
    
    // メソッド
    LightGBM_clf();
    LightGBM_clf(LightGBM_clf_param& param);
    void set_param(LightGBM_clf_param& param);
    void fit(const Eigen::MatrixXd &X, const Eigen::MatrixXd &y);
    Eigen::MatrixXd predict(const Eigen::MatrixXd& X_test);
    
    
    //テストクラスのフレンド宣言
    friend class LightGBMTest;
    FRIEND_TEST(LightGBMTest, count_label);
    FRIEND_TEST(LightGBMTest, set_leaf_value_on_root);
    FRIEND_TEST(LightGBMTest, init_0_generation_trees);
    FRIEND_TEST(LightGBMTest, make_cash_result);
    FRIEND_TEST(LightGBMTest, neg_gradient_cross_entropy);
    FRIEND_TEST(LightGBMTest, factor_neg_gradient);
    FRIEND_TEST(LightGBMTest, neg_variance_gain);
    FRIEND_TEST(LightGBMTest, loss_func_for_leaf);
    FRIEND_TEST(LightGBMTest, leaf_solver);
    FRIEND_TEST(LightGBMTest, random_sampling_without_replacement);
    FRIEND_TEST(LightGBMTest, GOS_sampling);
    FRIEND_TEST(LightGBMTest, fit);
    FRIEND_TEST(LightGBMTest, predict);
    
    
};

#endif /* LightGBM_hpp */
