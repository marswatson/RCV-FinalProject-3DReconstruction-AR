run('F:\各种课件\Rutgers\2015 fall\Robotics&Computer Vision\FinalProject\vlfeat-0.9.20-bin\vlfeat-0.9.20\toolbox\vl_setup');
run('F:\各种课件\Rutgers\2015 fall\Robotics&Computer Vision\FinalProject\vlfeat-0.9.20-bin\vlg\toolbox\vlg_setup');

Intrinsic = [5.9275660408893941e+002; 5.9275660408893941e+002; 326; 2.4450000000000000e+002];
K = [Intrinsic,Intrinsic,Intrinsic,Intrinsic,Intrinsic,Intrinsic];
T1 = [0 0 0];
T2 = [9.9092003241031690e-001 -8.1992265903755801e-002 -1.0655870541586158e-001];
T3 = [7.9731114589984398e-001 3.0959205312800765e-002 -5.8986754837429073e-002];
T4 = [7.3283711451169087e-001 -4.8900655815575393e-002 -3.7087799466634722e-002];
T5 = [5.6930043287928522e-001 -7.2652949860372601e-002 -2.9794481750940946e-002];
T6 = [4.5956948686787513e-001 -5.9809034789947889e-002 -3.7848467972029065e-002];
Te = [T1' T2' T3' T4' T5' T6'];

w1 = [0 0 0];
w2 = [8.3254064204464447e-006 -1.6660156654010391e-001 -3.4500036349706609e-002];
w3 = [2.5441229609926588e-002 -1.1184173942460905e-001 -1.4410919325647947e-002];
w4 = [-6.6656203478060007e-003 -1.1041522680123916e-001 -9.5790452765362015e-003];
w5 = [-4.3571814709137784e-003 -1.0400494551186729e-001  -3.1610727989168690e-002];
w6 = [1.1622423333243960e-002 -8.7657189499752736e-002 -2.5041660119586133e-002];
w = [w1',w2',w3',w4',w5',w6'];

Xe = importdata('points4D.txt');
Xe = Xe';

im1 = importdata('new1.txt');
im1 = im1';
z=find(~isnan(im1));
im1 = im1(z);
im1 = reshape(im1,3,238);

im2 = importdata('new2.txt');
im2 = im2';
z=find(~isnan(im2));
im2 = im2(z);
im2 = reshape(im2,3,238);

im3 = importdata('new3.txt');
im3 = im3';
z=find(~isnan(im3));
im3 = im3(z);
im3 = reshape(im3,3,238);

im4 = importdata('new4.txt');
im4 = im4';
z=find(~isnan(im4));
im4 = im4(z);
im4 = reshape(im4,3,238);

im5 = importdata('new5.txt');
im5 = im5';
z=find(~isnan(im5));
im5 = im5(z);
im5 = reshape(im5,3,238);

im6 = importdata('new6.txt');
im6 = im6';
z=find(~isnan(im6));
im6 = im6(z);
im6 = reshape(im6,3,238);

x = cat(3,im1,im2,im3,im4,im5,im6);

[K_,Te_,w_,Xe_,error_] = bundle_euclid_nomex( K, Te, w, Xe, x, 'verbose');

