# RL_INVEST
 system for agent choose fomula

#state.py
action: - 0 là không đầu tư
        - 1 là đầu tư theo công ty trả ra ở công thức 1
        - 2 là đầu tư theo công ty trả ra ở công thức 2
player_state:
        - 480 vị trí đầu (0-479) là top 20 công thức của 24 quý gần nhất của công thức 1, các quý càng ở vị trí cuối là càng gần với thời điểm xét
        - 480 vị trí tiếp (480-959) là top 20 công thức của 24 quý gần nhất của công thức 2, các quý càng ở vị trí cuối là càng gần với thời điểm xét
        - 2 vị trí tiếp theo (960-961) là gmean của thứ hạng được đánh giá của hành động trong mỗi turn của 2 người chơi(dải giá trị từ (1/3 - 1)). Đánh giá của bot nằm ở trước.
        -2 vị trí kế tiếp (962-963) là rank kết quả công thức ứng với giá trị không đầu tư được trả ra từ 2 công thức tại thời điểm được xét
        - vị trí cuối cùng (964) là tổng số cty (bao gồm cả NOT_INVEST) tại thời điểm đầu tư

env_state:
        - 2*ALL_QUARTER*TOP_COMP_PER_QUARTER vị trí đầu tiên (hiện tại là 2*1240) là rank profit của TOP_COMP_PER_QUARTER trong từng thời điểm của 2 công thức, ALL_QUARTER*TOP_COMP_PER_QUARTER vị trí đầu tiên là của công thức 1
        - 2*ALL_QUARTER vị trí tiếp theo là kết quả đánh giá action của agent và bot hệ thống, ALL_QUARTER vị trí đầu là của agent.
        - 2 vị trí kế tiếp là rank kết quả công thức ứng với giá trị không đầu tư được trả ra từ 2 công thức tại thời điểm được xét
        - 4 giá trị cuối là thời điểm đang xét, id_action, check hết game, tổng số cty quý hiện tại

cách sinh công thức:
        - sinh dấu của toán hạng con (0 là subtract, 1 là add)
        - với toán hạng con: 
                - sinh tập hợp biến ở tử số
                - sinh tập hợp biến ở mẫu số, sau đó thêm vào tập hợp số lượng biến A theo cách sinh bậc
        -id trong công thức:
        bắt đầu từ cột Market_Cap đến hết có id lần lượt là 0 -> number_variable 


#state_new.py
action: - 0 là không đầu tư
        - 1 là đầu tư theo công ty trả ra ở công thức 1
        - 2 là đầu tư theo công ty trả ra ở công thức 2
player_state:
        - 480 vị trí đầu (0-479) là top 20 công thức của 24 quý gần nhất của công thức 1, các quý càng ở vị trí cuối là càng gần với thời điểm xét
        - 480 vị trí tiếp (480-959) là top 20 công thức của 24 quý gần nhất của công thức 2, các quý càng ở vị trí cuối là càng gần với thời điểm xét
        - 2 vị trí tiếp theo (960-961) là gmean của thứ hạng được đánh giá của hành động trong mỗi turn của 2 người chơi(dải giá trị từ (1/3 - 1)). Đánh giá của bot nằm ở trước.
        -1 vị trí tiếp theo là rank của action ko đầu tư của quý trước (962)
        -2 vị trí kế tiếp (963-964) là rank kết quả công thức ứng với giá trị không đầu tư được trả ra từ 2 công thức tại thời điểm được xét
        - vị trí cuối cùng (965) là tổng số cty (bao gồm cả NOT_INVEST) tại thời điểm đầu tư

env_state:
        - 2*ALL_QUARTER*TOP_COMP_PER_QUARTER vị trí đầu tiên (hiện tại là 2*1240) là rank profit của TOP_COMP_PER_QUARTER trong từng thời điểm của 2 công thức, ALL_QUARTER*TOP_COMP_PER_QUARTER vị trí đầu tiên là của công thức 1
        - 2*ALL_QUARTER vị trí tiếp theo là kết quả đánh giá action của agent và bot hệ thống, ALL_QUARTER vị trí đầu là của agent.
        - 2 vị trí kế tiếp là rank kết quả công thức ứng với giá trị không đầu tư được trả ra từ 2 công thức tại thời điểm được xét
        - 5 giá trị cuối là: thời điểm đang xét, id_action, check hết game, tổng số cty quý hiện tại, rank theo lợi nhuận của ko đầu tư quý trước

cách sinh công thức:
        - sinh dấu của toán hạng con (0 là subtract, 1 là add)
        - với toán hạng con: 
                - sinh tập hợp biến ở tử số
                - sinh tập hợp biến ở mẫu số, sau đó thêm vào tập hợp số lượng biến A theo cách sinh bậc
        -id trong công thức:
        bắt đầu từ cột Market_Cap đến hết có id lần lượt là 0 -> number_variable 

