% Write Matlab matrix to alist formatted file

function writeAlist(filename, H)

fp = fopen(filename, 'w');
[M,N] = size(H);

cdegs = full(sum(H,2));
bdegs = full(sum(H,1));
max_cd = max(cdegs);
max_bd = max(bdegs);

fprintf(fp, '%d ', M);
fprintf(fp, '%d\n', N);
fprintf(fp, '%d ', max_cd);
fprintf(fp, '%d\n', max_bd);

fprintf(fp, '%d ', cdegs);
fprintf(fp, '\n');
fprintf(fp, '%d ', bdegs);

for i=1:M
    row = zeros(1,max_cd);
    row(1:cdegs(i)) = find(H(i,:));
    fprintf(fp, '%d ', row);
    fprintf(fp, '\n');
end

fclose(fp);
