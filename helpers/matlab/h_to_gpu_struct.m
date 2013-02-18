% This converts a Matlab parity check matrix to the structures required by the
% LDPC decoder library

function [Hcn_f, Hvn_f, llr_map, row_idx, col_idx] = h_to_gpu_struct(H)

[M, N] = size(H)

%Hi = sparse(zeros(M,N));

n_edges = sum(sum(H))
Hi = sparse(M,N, n_edges);
size(Hi)


Hcn = zeros(1,n_edges);
Hvn = zeros(1,n_edges);
llr_map = zeros(1, n_edges, 'int32'); %Mapping matrix elements to llr id
row_idx = zeros(1, M, 'int32');
col_idx = zeros(1, N, 'int32');


idx = 0;
rowlen = 0;

for (i = 1:M)
    idxrow = find(H(i,:));
    rowlen = length(idxrow);
    idxs = idx:idx+rowlen-1;
    Hi(i,idxrow) = idxs+1; %to make 0 index stand out from all zeroes in the matrix
    Hcn(idxs+1) = circshift(idxs, [1, -1]);
    row_idx(i) = idxs(1);
    idx = idx + rowlen;
end
rowlen=0;
    
for (i=1:N)
    idxcol = Hi(find(Hi(:,i)), i)';
    Hvn(idxcol) = circshift(idxcol-1, [1,-1]);
    llr_map(idxcol) = i-1;
    col_idx(i) = idxcol(1)-1;
end

Hcn_f = int32(Hcn); % C code expects integer arrays
Hvn_f = int32(Hvn);
