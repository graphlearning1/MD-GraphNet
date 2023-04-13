from run_model import *

max_spec_v = 0
max_kl_loss = 0
max_sim_v = 0
max_s_rec = 0
max_acc = 0

def text_create(name):
    desktop_path = "log20/"
    # 新创建的txt文件的存放路径
    full_path = desktop_path + name + '.txt'  # 也可以创建一个.doc的word文档
    file = open(full_path, 'w')

def run_dataset(dataset):
    max_spec_v = 0
    max_kl_loss = 0
    max_sim_v = 0
    max_s_rec = 0
    max_acc = 0

    run_model_per(0, 0, 0, 0, dataset, 20)


    text_create(dataset)
    output = sys.stdout
    outputfile = open("./log20/" + dataset + '5.txt', 'w')
    sys.stdout = outputfile
    try:
        for kl_loss in [0.5]:
            for i1 in [0.1]:
                for i2 in range(0, 11):
                    for i3 in range(0, 11):
                        # for i4 in range(4, 12):
                        sim_v = i1
                        # kl_loss = i2 / 10
                        spec_v = i2 / 10
                        s_rec = i3 / 5

                        acc_test = run_model_per(kl_loss, s_rec, sim_v, spec_v, dataset, 20) #cora,BlogCatalog, uai,acm

                        if acc_test >= max_acc:
                            max_spec_v = spec_v
                            max_sim_v = sim_v
                            max_s_rec = s_rec
                            max_acc = acc_test
                            max_kl_loss = kl_loss
                print(dataset, file=outputfile)
                print("__________________________", max_spec_v, max_kl_loss, max_sim_v, max_s_rec, max_acc,
                      file=outputfile)
                # print(max_spec_v, max_kl_loss, max_sim_v, max_s_rec, max_acc,
                #       file=outputfile)
                outputfile.close()  # close后才能看到写入的数据
                max_spec_v = 0
                max_kl_loss = 0
                max_sim_v = 0
                max_s_rec = 0
                max_acc = 0
    except:
        outputfile.close()

run_dataset('flickr')