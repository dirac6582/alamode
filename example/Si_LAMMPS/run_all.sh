#!/bin/bash

# Please modify the following paths appropriately
#export DYLD_LIBRARY_PATH=/Users/tadano/src/spglib/lib/:$DYLD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/Users/tadano/src/spglib/lib/:$LD_LIBRARY_PATH

# Binaries 
LAMMPS=${HOME}/src/lammps/_build/lmp
#LAMMPS=/usr/local/bin/lmp
ALAMODE_ROOT=${HOME}/src/alamode

# Generate displacement patterns

cat << EOF > si_alm0.in
&general
  PREFIX = si222
  MODE = suggest
  NAT = 64; NKD = 1
  KD = Si
/

&interaction
  NORDER = 2  # 1: harmonic, 2: cubic, ..
/

&cell
  20.406 # factor in Bohr unit
  1.0 0.0 0.0 # a1
  0.0 1.0 0.0 # a2
  0.0 0.0 1.0 # a3
/

&cutoff 
  Si-Si 7.3 7.3
/


&position
  1 0.0000000000000000 0.0000000000000000 0.0000000000000000   
  1 0.0000000000000000 0.0000000000000000 0.5000000000000000
  1 0.0000000000000000 0.2500000000000000 0.2500000000000000
  1 0.0000000000000000 0.2500000000000000 0.7500000000000000
  1 0.0000000000000000 0.5000000000000000 0.0000000000000000
  1 0.0000000000000000 0.5000000000000000 0.5000000000000000
  1 0.0000000000000000 0.7500000000000000 0.2500000000000000
  1 0.0000000000000000 0.7500000000000000 0.7500000000000000
  1 0.1250000000000000 0.1250000000000000 0.1250000000000000
  1 0.1250000000000000 0.1250000000000000 0.6250000000000000
  1 0.1250000000000000 0.3750000000000000 0.3750000000000000
  1 0.1250000000000000 0.3750000000000000 0.8750000000000000
  1 0.1250000000000000 0.6250000000000000 0.1250000000000000
  1 0.1250000000000000 0.6250000000000000 0.6250000000000000
  1 0.1250000000000000 0.8750000000000000 0.3750000000000000
  1 0.1250000000000000 0.8750000000000000 0.8750000000000000
  1 0.2500000000000000 0.0000000000000000 0.2500000000000000
  1 0.2500000000000000 0.0000000000000000 0.7500000000000000
  1 0.2500000000000000 0.2500000000000000 0.0000000000000000
  1 0.2500000000000000 0.2500000000000000 0.5000000000000000
  1 0.2500000000000000 0.5000000000000000 0.2500000000000000
  1 0.2500000000000000 0.5000000000000000 0.7500000000000000
  1 0.2500000000000000 0.7500000000000000 0.0000000000000000
  1 0.2500000000000000 0.7500000000000000 0.5000000000000000
  1 0.3750000000000000 0.1250000000000000 0.3750000000000000
  1 0.3750000000000000 0.1250000000000000 0.8750000000000000
  1 0.3750000000000000 0.3750000000000000 0.1250000000000000
  1 0.3750000000000000 0.3750000000000000 0.6250000000000000
  1 0.3750000000000000 0.6250000000000000 0.3750000000000000
  1 0.3750000000000000 0.6250000000000000 0.8750000000000000
  1 0.3750000000000000 0.8750000000000000 0.1250000000000000
  1 0.3750000000000000 0.8750000000000000 0.6250000000000000
  1 0.5000000000000000 0.0000000000000000 0.0000000000000000
  1 0.5000000000000000 0.0000000000000000 0.5000000000000000
  1 0.5000000000000000 0.2500000000000000 0.2500000000000000
  1 0.5000000000000000 0.2500000000000000 0.7500000000000000
  1 0.5000000000000000 0.5000000000000000 0.0000000000000000
  1 0.5000000000000000 0.5000000000000000 0.5000000000000000
  1 0.5000000000000000 0.7500000000000000 0.2500000000000000
  1 0.5000000000000000 0.7500000000000000 0.7500000000000000
  1 0.6250000000000000 0.1250000000000000 0.1250000000000000
  1 0.6250000000000000 0.1250000000000000 0.6250000000000000
  1 0.6250000000000000 0.3750000000000000 0.3750000000000000
  1 0.6250000000000000 0.3750000000000000 0.8750000000000000
  1 0.6250000000000000 0.6250000000000000 0.1250000000000000
  1 0.6250000000000000 0.6250000000000000 0.6250000000000000
  1 0.6250000000000000 0.8750000000000000 0.3750000000000000
  1 0.6250000000000000 0.8750000000000000 0.8750000000000000
  1 0.7500000000000000 0.0000000000000000 0.2500000000000000
  1 0.7500000000000000 0.0000000000000000 0.7500000000000000
  1 0.7500000000000000 0.2500000000000000 0.0000000000000000
  1 0.7500000000000000 0.2500000000000000 0.5000000000000000
  1 0.7500000000000000 0.5000000000000000 0.2500000000000000
  1 0.7500000000000000 0.5000000000000000 0.7500000000000000
  1 0.7500000000000000 0.7500000000000000 0.0000000000000000
  1 0.7500000000000000 0.7500000000000000 0.5000000000000000
  1 0.8750000000000000 0.1250000000000000 0.3750000000000000
  1 0.8750000000000000 0.1250000000000000 0.8750000000000000
  1 0.8750000000000000 0.3750000000000000 0.1250000000000000
  1 0.8750000000000000 0.3750000000000000 0.6250000000000000
  1 0.8750000000000000 0.6250000000000000 0.3750000000000000
  1 0.8750000000000000 0.6250000000000000 0.8750000000000000
  1 0.8750000000000000 0.8750000000000000 0.1250000000000000
  1 0.8750000000000000 0.8750000000000000 0.6250000000000000
/

EOF

${ALAMODE_ROOT}/alm/alm si_alm0.in > alm.log


# Generate structure files of LAMMPS
mkdir displace; cd displace/

python ${ALAMODE_ROOT}/tools/displace.py --LAMMPS ../Si222.lammps --prefix harm --mag 0.01 -pf ../si222.pattern_HARMONIC >> run.log
python ${ALAMODE_ROOT}/tools/displace.py --LAMMPS ../Si222.lammps --prefix cubic --mag 0.04 -pf ../si222.pattern_ANHARM3 >> run.log

cp ../Si.sw .
cp ../in.sw .

# Run LAMMPS
for ((i=1; i<=1; i++))
do
   cp harm${i}.lammps tmp.lammps
   $LAMMPS < in.sw >> run.log
   mv XFSET XFSET.harm${i}
done

for ((i=1; i<=20; i++))
do
   suffix=`echo ${i} | awk '{printf("%02d", $1)}'`
   cp cubic${suffix}.lammps tmp.lammps
   $LAMMPS < in.sw >> run.log
   mv XFSET XFSET.cubic${suffix}
done

# Collect data
python ${ALAMODE_ROOT}/tools/extract.py --LAMMPS ../Si222.lammps XFSET.harm* > DFSET_harmonic
python ${ALAMODE_ROOT}/tools/extract.py --LAMMPS ../Si222.lammps XFSET.cubic* > DFSET_cubic

cd ../

# Extract harmonic force constants
cat << EOF > si_alm1.in
&general
  PREFIX = si222_harm
  MODE = optimize
  NAT = 64; NKD = 1
  KD = Si
/

&optimize
 DFSET = displace/DFSET_harmonic
/

&interaction
  NORDER = 1  # 1: harmonic, 2: cubic, ..
/

&cell
  20.406 # factor in Bohr unit
  1.0 0.0 0.0 # a1
  0.0 1.0 0.0 # a2
  0.0 0.0 1.0 # a3
/

&cutoff 
  Si-Si 7.3 7.3
/


&position
  1 0.0000000000000000 0.0000000000000000 0.0000000000000000   
  1 0.0000000000000000 0.0000000000000000 0.5000000000000000
  1 0.0000000000000000 0.2500000000000000 0.2500000000000000
  1 0.0000000000000000 0.2500000000000000 0.7500000000000000
  1 0.0000000000000000 0.5000000000000000 0.0000000000000000
  1 0.0000000000000000 0.5000000000000000 0.5000000000000000
  1 0.0000000000000000 0.7500000000000000 0.2500000000000000
  1 0.0000000000000000 0.7500000000000000 0.7500000000000000
  1 0.1250000000000000 0.1250000000000000 0.1250000000000000
  1 0.1250000000000000 0.1250000000000000 0.6250000000000000
  1 0.1250000000000000 0.3750000000000000 0.3750000000000000
  1 0.1250000000000000 0.3750000000000000 0.8750000000000000
  1 0.1250000000000000 0.6250000000000000 0.1250000000000000
  1 0.1250000000000000 0.6250000000000000 0.6250000000000000
  1 0.1250000000000000 0.8750000000000000 0.3750000000000000
  1 0.1250000000000000 0.8750000000000000 0.8750000000000000
  1 0.2500000000000000 0.0000000000000000 0.2500000000000000
  1 0.2500000000000000 0.0000000000000000 0.7500000000000000
  1 0.2500000000000000 0.2500000000000000 0.0000000000000000
  1 0.2500000000000000 0.2500000000000000 0.5000000000000000
  1 0.2500000000000000 0.5000000000000000 0.2500000000000000
  1 0.2500000000000000 0.5000000000000000 0.7500000000000000
  1 0.2500000000000000 0.7500000000000000 0.0000000000000000
  1 0.2500000000000000 0.7500000000000000 0.5000000000000000
  1 0.3750000000000000 0.1250000000000000 0.3750000000000000
  1 0.3750000000000000 0.1250000000000000 0.8750000000000000
  1 0.3750000000000000 0.3750000000000000 0.1250000000000000
  1 0.3750000000000000 0.3750000000000000 0.6250000000000000
  1 0.3750000000000000 0.6250000000000000 0.3750000000000000
  1 0.3750000000000000 0.6250000000000000 0.8750000000000000
  1 0.3750000000000000 0.8750000000000000 0.1250000000000000
  1 0.3750000000000000 0.8750000000000000 0.6250000000000000
  1 0.5000000000000000 0.0000000000000000 0.0000000000000000
  1 0.5000000000000000 0.0000000000000000 0.5000000000000000
  1 0.5000000000000000 0.2500000000000000 0.2500000000000000
  1 0.5000000000000000 0.2500000000000000 0.7500000000000000
  1 0.5000000000000000 0.5000000000000000 0.0000000000000000
  1 0.5000000000000000 0.5000000000000000 0.5000000000000000
  1 0.5000000000000000 0.7500000000000000 0.2500000000000000
  1 0.5000000000000000 0.7500000000000000 0.7500000000000000
  1 0.6250000000000000 0.1250000000000000 0.1250000000000000
  1 0.6250000000000000 0.1250000000000000 0.6250000000000000
  1 0.6250000000000000 0.3750000000000000 0.3750000000000000
  1 0.6250000000000000 0.3750000000000000 0.8750000000000000
  1 0.6250000000000000 0.6250000000000000 0.1250000000000000
  1 0.6250000000000000 0.6250000000000000 0.6250000000000000
  1 0.6250000000000000 0.8750000000000000 0.3750000000000000
  1 0.6250000000000000 0.8750000000000000 0.8750000000000000
  1 0.7500000000000000 0.0000000000000000 0.2500000000000000
  1 0.7500000000000000 0.0000000000000000 0.7500000000000000
  1 0.7500000000000000 0.2500000000000000 0.0000000000000000
  1 0.7500000000000000 0.2500000000000000 0.5000000000000000
  1 0.7500000000000000 0.5000000000000000 0.2500000000000000
  1 0.7500000000000000 0.5000000000000000 0.7500000000000000
  1 0.7500000000000000 0.7500000000000000 0.0000000000000000
  1 0.7500000000000000 0.7500000000000000 0.5000000000000000
  1 0.8750000000000000 0.1250000000000000 0.3750000000000000
  1 0.8750000000000000 0.1250000000000000 0.8750000000000000
  1 0.8750000000000000 0.3750000000000000 0.1250000000000000
  1 0.8750000000000000 0.3750000000000000 0.6250000000000000
  1 0.8750000000000000 0.6250000000000000 0.3750000000000000
  1 0.8750000000000000 0.6250000000000000 0.8750000000000000
  1 0.8750000000000000 0.8750000000000000 0.1250000000000000
  1 0.8750000000000000 0.8750000000000000 0.6250000000000000
/

EOF
${ALAMODE_ROOT}/alm/alm si_alm1.in >> alm.log

# Extract cubic force constants
cat << EOF > si_alm2.in
&general
  PREFIX = si222_cubic
  MODE = optimize
  NAT = 64; NKD = 1
  KD = Si
/

&optimize
 DFSET = displace/DFSET_cubic
 FC2XML = si222_harm.xml
/

&interaction
  NORDER = 2  # 1: harmonic, 2: cubic, ..
/

&cell
  20.406 # factor in Bohr unit
  1.0 0.0 0.0 # a1
  0.0 1.0 0.0 # a2
  0.0 0.0 1.0 # a3
/

&cutoff 
  Si-Si 7.3 7.3
/


&position
  1 0.0000000000000000 0.0000000000000000 0.0000000000000000   
  1 0.0000000000000000 0.0000000000000000 0.5000000000000000
  1 0.0000000000000000 0.2500000000000000 0.2500000000000000
  1 0.0000000000000000 0.2500000000000000 0.7500000000000000
  1 0.0000000000000000 0.5000000000000000 0.0000000000000000
  1 0.0000000000000000 0.5000000000000000 0.5000000000000000
  1 0.0000000000000000 0.7500000000000000 0.2500000000000000
  1 0.0000000000000000 0.7500000000000000 0.7500000000000000
  1 0.1250000000000000 0.1250000000000000 0.1250000000000000
  1 0.1250000000000000 0.1250000000000000 0.6250000000000000
  1 0.1250000000000000 0.3750000000000000 0.3750000000000000
  1 0.1250000000000000 0.3750000000000000 0.8750000000000000
  1 0.1250000000000000 0.6250000000000000 0.1250000000000000
  1 0.1250000000000000 0.6250000000000000 0.6250000000000000
  1 0.1250000000000000 0.8750000000000000 0.3750000000000000
  1 0.1250000000000000 0.8750000000000000 0.8750000000000000
  1 0.2500000000000000 0.0000000000000000 0.2500000000000000
  1 0.2500000000000000 0.0000000000000000 0.7500000000000000
  1 0.2500000000000000 0.2500000000000000 0.0000000000000000
  1 0.2500000000000000 0.2500000000000000 0.5000000000000000
  1 0.2500000000000000 0.5000000000000000 0.2500000000000000
  1 0.2500000000000000 0.5000000000000000 0.7500000000000000
  1 0.2500000000000000 0.7500000000000000 0.0000000000000000
  1 0.2500000000000000 0.7500000000000000 0.5000000000000000
  1 0.3750000000000000 0.1250000000000000 0.3750000000000000
  1 0.3750000000000000 0.1250000000000000 0.8750000000000000
  1 0.3750000000000000 0.3750000000000000 0.1250000000000000
  1 0.3750000000000000 0.3750000000000000 0.6250000000000000
  1 0.3750000000000000 0.6250000000000000 0.3750000000000000
  1 0.3750000000000000 0.6250000000000000 0.8750000000000000
  1 0.3750000000000000 0.8750000000000000 0.1250000000000000
  1 0.3750000000000000 0.8750000000000000 0.6250000000000000
  1 0.5000000000000000 0.0000000000000000 0.0000000000000000
  1 0.5000000000000000 0.0000000000000000 0.5000000000000000
  1 0.5000000000000000 0.2500000000000000 0.2500000000000000
  1 0.5000000000000000 0.2500000000000000 0.7500000000000000
  1 0.5000000000000000 0.5000000000000000 0.0000000000000000
  1 0.5000000000000000 0.5000000000000000 0.5000000000000000
  1 0.5000000000000000 0.7500000000000000 0.2500000000000000
  1 0.5000000000000000 0.7500000000000000 0.7500000000000000
  1 0.6250000000000000 0.1250000000000000 0.1250000000000000
  1 0.6250000000000000 0.1250000000000000 0.6250000000000000
  1 0.6250000000000000 0.3750000000000000 0.3750000000000000
  1 0.6250000000000000 0.3750000000000000 0.8750000000000000
  1 0.6250000000000000 0.6250000000000000 0.1250000000000000
  1 0.6250000000000000 0.6250000000000000 0.6250000000000000
  1 0.6250000000000000 0.8750000000000000 0.3750000000000000
  1 0.6250000000000000 0.8750000000000000 0.8750000000000000
  1 0.7500000000000000 0.0000000000000000 0.2500000000000000
  1 0.7500000000000000 0.0000000000000000 0.7500000000000000
  1 0.7500000000000000 0.2500000000000000 0.0000000000000000
  1 0.7500000000000000 0.2500000000000000 0.5000000000000000
  1 0.7500000000000000 0.5000000000000000 0.2500000000000000
  1 0.7500000000000000 0.5000000000000000 0.7500000000000000
  1 0.7500000000000000 0.7500000000000000 0.0000000000000000
  1 0.7500000000000000 0.7500000000000000 0.5000000000000000
  1 0.8750000000000000 0.1250000000000000 0.3750000000000000
  1 0.8750000000000000 0.1250000000000000 0.8750000000000000
  1 0.8750000000000000 0.3750000000000000 0.1250000000000000
  1 0.8750000000000000 0.3750000000000000 0.6250000000000000
  1 0.8750000000000000 0.6250000000000000 0.3750000000000000
  1 0.8750000000000000 0.6250000000000000 0.8750000000000000
  1 0.8750000000000000 0.8750000000000000 0.1250000000000000
  1 0.8750000000000000 0.8750000000000000 0.6250000000000000
/

EOF
${ALAMODE_ROOT}/alm/alm si_alm2.in >> alm.log

# Phonon dispersion
cat << EOF > phband.in
&general
  PREFIX = si222
  MODE = phonons
  FCSXML =si222_harm.xml

  NKD = 1; KD = Si
  MASS = 28.0855
/

&cell
  10.203
  0.0 0.5 0.5
  0.5 0.0 0.5
  0.5 0.5 0.0
/

&kpoint
  1  # KPMODE = 1: line mode
  G 0.0 0.0 0.0 X 0.5 0.5 0.0 51
  X 0.5 0.5 1.0 G 0.0 0.0 0.0 51
  G 0.0 0.0 0.0 L 0.5 0.5 0.5 51
/

EOF

${ALAMODE_ROOT}/anphon/anphon phband.in > phband.log

# Thermal conductivity
cat << EOF > RTA.in
&general
  PREFIX = si222_10
  MODE = RTA
  FCSXML = si222_cubic.xml

  NKD = 1; KD = Si
  MASS = 28.0855
/

&cell
  10.203
  0.0 0.5 0.5
  0.5 0.0 0.5
  0.5 0.5 0.0
/

&kpoint
  2
  10 10 10
/

EOF

${ALAMODE_ROOT}/anphon/anphon RTA.in > RTA.log
