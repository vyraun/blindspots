wget https://github.com/HendrikStrobelt/Seq2Seq-Vis/archive/master.zip
unzip master.zip
cd Seq2Seq-Vis-master
source setup_cpu.sh
source activate s2sv
pip install torchtext=0.2.3

wget https://github.com/sebastianGehrmann/OpenNMT-py/archive/states_in_translation.zip
unzip states_in_translation.zip
mv OpenNMT-py-states_in_translation OpenNMT-py
cd OpenNMT-py
python setup.py install
pip install torchtext
cd ..
