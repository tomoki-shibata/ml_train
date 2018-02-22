//
//  tree.hpp
//  ml_codes
//
//  Created by 柴田 智喜 on 2018/01/28.
//  Copyright © 2018年 柴田 智喜. All rights reserved.
//

#ifndef tree_hpp
#define tree_hpp

#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <eigen3/Eigen/Dense>
#include <map>

class Node{
public:
    Node* parent_node;
    Node* more_node;
    Node* less_node;
    
    double split_value;
    double leaf_value;
    
    std::string path;
    int depth;
    int feature_idx;
    
    Node();
    
};

class Tree{
    
    //private:
public:
    Node* root_node;
    Node* temp_node;
    std::vector<Node*> leaf_list;
    std::vector<Node*> node_list;
    
public:
    //コンストラクタ関連
    Tree();
    void set_root_node();
    
    //デストラクタ・ノード削除
    ~Tree();
    void del_this_node();
    void del_all_nodes();
    void leaf_list_refresh();
    
    // 有無の確認
    bool is_parent_node();
    bool is_more_node();
    bool is_less_node();
    
    // ノードの追加
    void add_more_node();
    void add_less_node();
    
    //参照ノードの移動
    void to_parent_node();
    void to_more_node();
    void to_less_node();
    void to_root_node();
    void to_leaf_node();
    
    // 参照ノードの取得
    Node* get_node();
    //葉ノードリストの取得
    std::vector<Node*> get_leaf_list();
    
};




#endif /* tree_hpp */
