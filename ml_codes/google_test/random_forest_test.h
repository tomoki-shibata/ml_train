//
//  random_forest_test.h
//  ml_codes
//
//  Created by 柴田 智喜 on 2018/02/03.
//  Copyright © 2018年 柴田 智喜. All rights reserved.
//

#ifndef random_forest_test_h
#define random_forest_test_h
#include "gtest/gtest.h"
#include "random_forest.hpp"

using namespace std;

namespace{
    class RandomForestTest : public ::testing::Test{
    protected:
        RandomForestTest(){}
        virtual ~RandomForestTest(){}
        virtual void SetUp(){}
        virtual void TearDown(){}
    };
}

struct Rand_for_test:Random_Forest_clf::Rand{
    vector<int> row_sampling(int sample_num, int rows){
        
        vector<int> temp;
        for(int i = rows-1; i >= 0; i--){
            temp.push_back(i);
            if(temp.size() >= sample_num) break;
            temp.push_back(i);
            if(temp.size() >= sample_num) break;
            
        }

        return temp;
    };
    
    vector<int> col_sampling(int sample_num, int cols){
        vector<int> temp;
        for(int i = cols-1; i >= 0; i--){
            temp.push_back(i);
            if(temp.size() >= sample_num){
                break;
            }
        }
        
        return temp;
    };
};




TEST_F(RandomForestTest, row_sampling)
{
    Random_Forest_clf::Random_Forest_clf_param param;
    param.random_seed = 0;
    Eigen::MatrixXd X_train = Eigen::MatrixXd(10,4);

    Random_Forest_clf forest(param);
    forest.rand->mt_rand.seed(param.random_seed);
    forest.X_train = &X_train;
    
    for (int i = 0;i<100;++i){
        vector<int> row_samples = forest.rand->row_sampling(20,(int)forest.X_train->rows());
        sort(row_samples.begin(), row_samples.end());
        
        EXPECT_EQ(20,row_samples.size());
        EXPECT_EQ(true,0 <= (*row_samples.begin()));
        EXPECT_EQ(true,9 >= (*row_samples.rbegin()));
    }
    
    
    param.random_seed = 0;
    X_train = Eigen::MatrixXd(100,4);
    
    forest.rand->mt_rand.seed(param.random_seed);
    forest.X_train = &X_train;
    
    for (int i = 0;i<100;++i){
        vector<int> row_samples = forest.rand->row_sampling(100,(int)forest.X_train->rows());
        sort(row_samples.begin(), row_samples.end());
        
        EXPECT_EQ(100,row_samples.size());
        EXPECT_EQ(true,0 <= (*row_samples.begin()));
        EXPECT_EQ(true,99 >= (*row_samples.rbegin()));
    }
}


TEST_F(RandomForestTest, col_sampling)
{
    Random_Forest_clf::Random_Forest_clf_param param;
    param.random_seed = 0;
    Eigen::MatrixXd X_train = Eigen::MatrixXd(10,10);
    
    Random_Forest_clf forest(param);
    forest.rand->mt_rand.seed(param.random_seed);
    forest.X_train = &X_train;
    
    for (int i = 0;i<100;++i){
        vector<int> col_samples = forest.rand->col_sampling(5,(int)forest.X_train->cols());
        sort(col_samples.begin(), col_samples.end());
        map<double,int> check;
        for(const auto &iter :col_samples){
            check[iter] += 1;
        }
        
        EXPECT_EQ(5,col_samples.size());
        EXPECT_EQ(true,0 <= (*col_samples.begin()));
        EXPECT_EQ(true,9 >= (*col_samples.rbegin()));

        
        for (const auto &iter:check){
            EXPECT_EQ(1,iter.second);
        }
    }
    
    param.random_seed = 0;
    X_train = Eigen::MatrixXd(10,100);
    
    forest.rand->mt_rand.seed(param.random_seed);
    forest.X_train = &X_train;
    
    for (int i = 0;i<100;++i){
        vector<int> col_samples = forest.rand->col_sampling(100,(int)forest.X_train->cols());
        sort(col_samples.begin(), col_samples.end());
        map<double,int> check;
        for(const auto &iter :col_samples){
            check[iter] += 1;
        }
        
        EXPECT_EQ(100,col_samples.size());
        EXPECT_EQ(true,0 <= (*col_samples.begin()));
        EXPECT_EQ(true,99 >= (*col_samples.rbegin()));
        
        for (const auto &iter:check){
            EXPECT_EQ(1,iter.second);
        }
    }
    
}


TEST_F(RandomForestTest, make_split)
{
    //パラメータ設定
    Random_Forest_clf::Random_Forest_clf_param param;
    param.random_seed = 12;
    param.feature_sample_num = 3;
    
    // 変数宣言
    Eigen::MatrixXd X_train;
    Eigen::MatrixXd y_train;
    vector<int> data_idx;
    Random_Forest_clf::RF_CLF_tree::Tree_Split split;
    Random_Forest_clf::RF_CLF_tree rf_clf_tree;
    
    // 各種設定
    Random_Forest_clf forest(param);
    
    Rand_for_test test_rand;
    forest.rand = &test_rand;
    forest.rand->mt_rand.seed(param.random_seed);
    rf_clf_tree.set_forest(&forest);
    
    //テスト1
    X_train = Eigen::MatrixXd(8,5);
    X_train <<
    0,0,1,0,1,
    0,0,1,0,1,
    0,0,1,0,1,
    0,0,0,0,1,
    1,0,0,0,1,
    1,0,0,0,1,
    1,0,0,1,1,
    1,0,0,1,1;
    
    y_train = Eigen::MatrixXd(8,1);
    y_train << 1,1,1,1,0,0,0,0;
    
    data_idx = vector<int>{0,1,2,3,4,5,6,7};
    
    rf_clf_tree.X_train = &X_train;
    rf_clf_tree.y_train = &y_train;
    split = rf_clf_tree.make_split(data_idx);
    
    EXPECT_EQ(2, split.feature_idx);
    EXPECT_EQ(3, split.more_idx.size());
    EXPECT_EQ(5, split.less_idx.size());
    EXPECT_EQ(1, split.split_value);
    
    
    //テスト2
    data_idx = vector<int>{0,1,2,3,6,7};
    
    split = rf_clf_tree.make_split(data_idx);
    
    EXPECT_EQ(3, split.feature_idx);
    EXPECT_EQ(2, split.more_idx.size());
    EXPECT_EQ(4, split.less_idx.size());
    EXPECT_EQ(1, split.split_value);
    
    //テスト3
    X_train <<
    0,0,7,   0,   1,
    0,0,5, 1.9,   1,
    0,0,6,-2.6,   1,
    0,0,3, 0.8,   1,
    1,0,3, 1.2,   1,
    1,0,3,-1.3,   1,
    1,0,4, 2.0,   1,
    1,0,2, 4.1,   1;
    
    y_train << 1,1,1,1,0,0,0,0;
    
    data_idx = vector<int>{0,1,2,3,4,5,6,7};
    
    rf_clf_tree.X_train = &X_train;
    rf_clf_tree.y_train = &y_train;
    split = rf_clf_tree.make_split(data_idx);
    
    EXPECT_EQ(2, split.feature_idx);
    EXPECT_EQ(3, split.more_idx.size());
    EXPECT_EQ(5, split.less_idx.size());
    EXPECT_EQ(5, split.split_value);
    
}


TEST_F(RandomForestTest, fit){
    
    //パラメータ設定
    Random_Forest_clf::Random_Forest_clf_param param;
    param.random_seed = 0;
    param.feature_sample_num = 4;
    param.estimators_num = 10;
    param.tree_param.min_samples_split = 2;
    param.tree_param.max_depth = 4;
    
    // 変数宣言
    Eigen::MatrixXd X_train;
    Eigen::MatrixXd y_train;
    vector<int> data_idx;
    
    // 乱数のテスト用設定
    Random_Forest_clf forest(param);
    Rand_for_test test_rand;
    forest.rand = &test_rand;
    
    X_train = Eigen::MatrixXd(16,7);
    X_train <<
    0,  0,  0,  1,  0,   1,   0,
    0,  0,  0,  2,  0,   1,   0,
    0,  0,  0,  3,  0,   1,   0,
    0,  0,  0,  4,  0,   1,   0,
    0,  0,  0,  5,  0,  -1,   0,
    0,  0,  0,  6,  0,  -1,   0,
    0,  0,  0,  7,  0,  -1,   0,
    0,  0,  0,  8,  0,   0,   0,
    0,  1,  0,  7,  1,   0,   1,
    0,  1,  0,  5,  1, 1.9,   1,
    0,  1,  0,  6,  1,-2.6,   1,
    0,  1,  0,  3,  1, 0.8,   1,
    0,  0,  1,  3,  1, 1.2,   1,
    0,  0,  1,  3,  1,-1.3,   1,
    0,  0,  2,  4,  1, 2.0,   1,
    0,  0,  1,  2,  1, 4.1,   1;
    
    y_train = Eigen::MatrixXd(16,1);
    y_train << 13,13,13,-2,-2,-2,7,8,
               3,3,3,3,-1,-1,-2,-1;
    
    forest.fit(X_train,y_train);
    EXPECT_EQ(2,forest.forest[0].param.min_samples_split);
    EXPECT_EQ(4,forest.forest[0].param.max_depth);
    
    
    for(auto& iter: forest.forest){
        vector<Node*>leaf_list = iter.tree.get_leaf_list();
        EXPECT_EQ("/dep_1,f_idx_3,spt_5,more/leaf_val->3",leaf_list[0]->path);
        EXPECT_EQ("/dep_1,f_idx_3,spt_5,less/dep_2,f_idx_3,spt_4,more/leaf_val->-2",leaf_list[1]->path);
        EXPECT_EQ("/dep_1,f_idx_3,spt_5,less/dep_2,f_idx_3,spt_4,less/dep_3,f_idx_5,spt_1.2,more/leaf_val->-1",leaf_list[2]->path);
        EXPECT_EQ("/dep_1,f_idx_3,spt_5,less/dep_2,f_idx_3,spt_4,less/dep_3,f_idx_5,spt_1.2,less/dep_4,f_idx_5,spt_0.8,more/leaf_val->3",leaf_list[3]->path);
        EXPECT_EQ("/dep_1,f_idx_3,spt_5,less/dep_2,f_idx_3,spt_4,less/dep_3,f_idx_5,spt_1.2,less/dep_4,f_idx_5,spt_0.8,less/leaf_val->-1",leaf_list[4]->path);
    }
    
    
    param.tree_param.min_samples_split = 3;
    param.tree_param.max_depth = 2;
    forest.set_param(param);
    forest.fit(X_train,y_train);
    
    EXPECT_EQ(3,forest.forest[0].param.min_samples_split);
    EXPECT_EQ(2,forest.forest[0].param.max_depth);
    
    
    for(auto& iter: forest.forest){
        vector<Node*>leaf_list = iter.tree.get_leaf_list();
        EXPECT_EQ("/dep_1,f_idx_3,spt_5,more/leaf_val->3",leaf_list[0]->path);
        EXPECT_EQ("/dep_1,f_idx_3,spt_5,less/dep_2,f_idx_3,spt_4,more/leaf_val->-2",leaf_list[1]->path);
        EXPECT_EQ("/dep_1,f_idx_3,spt_5,less/dep_2,f_idx_3,spt_4,less/leaf_val->-1",leaf_list[2]->path);
    }

    
}

TEST_F(RandomForestTest, predict){
    
    // 変数宣言
    Eigen::MatrixXd X_train;
    Eigen::MatrixXd y_train;
    vector<int> data_idx;
    Random_Forest_clf::Random_Forest_clf_param param;
    Random_Forest_clf forest;
    
    //パラメータ設定
    param.random_seed = 0;
    param.feature_sample_num = 4;
    param.estimators_num = 10;
    param.tree_param.min_samples_split = 3;
    param.tree_param.max_depth = 2;
    forest.set_param(param);
    
    // 乱数のテスト用設定
    Rand_for_test test_rand;
    forest.rand = &test_rand;
    
    X_train = Eigen::MatrixXd(16,7);
    X_train <<
    0,  0,  0,  1,  0,   1,   0,
    0,  0,  0,  2,  0,   1,   0,
    0,  0,  0,  3,  0,   1,   0,
    0,  0,  0,  4,  0,   1,   0,
    0,  0,  0,  5,  0,  -1,   0,
    0,  0,  0,  6,  0,  -1,   0,
    0,  0,  0,  7,  0,  -1,   0,
    0,  0,  0,  8,  0,   0,   0,
    0,  1,  0,  7,  1,   0,   1,
    0,  1,  0,  5,  1, 1.9,   1,
    0,  1,  0,  6,  1,-2.6,   1,
    0,  1,  0,  3,  1, 0.8,   1,
    0,  0,  1,  3,  1, 1.2,   1,
    0,  0,  1,  3,  1,-1.3,   1,
    0,  0,  2,  4,  1, 2.0,   1,
    0,  0,  1,  2,  1, 4.1,   1;
    
    y_train = Eigen::MatrixXd(16,1);
    y_train << 13,13,13,-2,-2,-2,7,8,
    3,3,3,3,-1,-1,-2,-1;
    
    forest.fit(X_train,y_train);
    
    Eigen::MatrixXd X_test(12,7);
    X_test <<
    0,  0,  0,  7,   1,   0,   1,
    0,  0,  0,  9,   12,   -356,   4.56,
    0,  0,  0,  5,   1, 1.9,   1,
    0,  0,  0,  5.75,   -367, 1.989,   -1,
    0,  0,  0,  4,   1, 2.0,   1,
    0,  0,  0,  4.999,   1, 2.2,   1,
    0,  0,  0,  3,   1, 1.2,   1,
    0,  0,  0,  2.9,   1, 200,   1,
    0,  0,  0,  3,   1, 0.8,   1,
    0,  0,  0,  3.999,   -1, 1.19,   1,
    0,  0,  0,  3,   1,-1.3,   1,
    0,  0,  0,  3.999,   1234, 0.7999,   -99999999;
    
    Eigen::MatrixXd y_test(12,1);
    y_test << 3,3,3,3,-2,-2,-1,-1,-1,-1,-1,-1;
    
    Eigen::MatrixXd result = forest.predict(X_test);
    EXPECT_EQ(y_test, forest.predict(X_test));
    
    // テスト2
    param.tree_param.min_samples_split = 2;
    param.tree_param.max_depth = 4;
    forest.set_param(param);
    forest.fit(X_train,y_train);
    y_test <<  3,3,3,3,-2,-2,-1,-1,3,3,-1,-1;
    result = forest.predict(X_test);
    
    EXPECT_EQ(y_test, forest.predict(X_test));

    // テスト3
    param.tree_param.min_samples_split = 2;
    param.tree_param.max_depth = 3;
    param.estimators_num = 300;
    param.random_seed = 16;
    forest.set_param(param);
    forest.rand = &forest.rand_instance;
    
    X_train = Eigen::MatrixXd(8,7);
    X_train <<
    0,  1,  0,  7,  1,   0,   1,
    0,  1,  0,  5,  1, 1.9,   1,
    0,  1,  0,  6,  1,-2.6,   1,
    0,  1,  0,  3,  1, 0.8,   1,
    0,  1,  0,  3,  1, 1.2,   1,
    0,  1,  0,  3,  1,-1.3,   1,
    0,  1,  0,  4,  1, 2.0,   1,
    0,  1,  0,  2,  1, 4.1,   1;
    
    y_train = Eigen::MatrixXd(8,1);
    y_train << 3,3,3,3,1,1,2,1;

    forest.fit(X_train,y_train);
    
    X_test <<
    0,  0,  0,  7,   1,   0,   1,
    0,  0,  0,  9,   12,   -356,   4.56,
    0,  0,  0,  5,   1, 1.9,   1,
    0,  0,  0,  5.75,   -367, 1.989,   -1,
    0,  0,  0,  4,   1, 2.0,   1,
    0,  0,  0,  4.999,   1, 2.2,   1,
    0,  0,  0,  3,   1, 1.2,   1,
    0,  0,  0,  2.9,   1, 200,   1,
    0,  0,  0,  3,   1, 0.8,   1,
    0,  0,  0,  3.999,   -1, 1.19,   1,
    0,  0,  0,  3,   1,-1.3,   1,
    0,  0,  0,  3.999,   1234, 0.7999,   -99999999;
    
    y_test <<  3,3,3,3,2,2,1,1,3,3,1,1;
    
    result = forest.predict(X_test);
    
    EXPECT_EQ(y_test, result);
    
    
}

#endif /* random_forest_test_h */
