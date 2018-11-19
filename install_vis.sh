wget https://github.com/HendrikStrobelt/Seq2Seq-Vis/archive/master.zip
unzip master.zip
cd Seq2Seq-Vis-master
source setup_cpu.sh
source activate s2sv

wget https://github.com/sebastianGehrmann/OpenNMT-py/archive/states_in_translation.zip
cd OpenNMT-py/
python setup.py install
pip install torchtext
cd ..
