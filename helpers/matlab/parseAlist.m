% Convert alist file (name) to a sparse matlab matrix

function H = parseAlist(name)

fp = fopen(name, 'r');
temp = fscanf(fp, '%d', 4);
M = temp(1)
N = temp(2)
K = N-M

max_cd = temp(3);
max_bd = temp(4);

H = sparse(M,N);

cdegs = fscanf(fp, '%d', M);
bdegs = fscanf(fp, '%d', N);

for i=1:M
    row = fscanf(fp, '%d', max_cd);
    H(i,row(1:cdegs(i))) = 1;
end

fclose(fp);
    
    

