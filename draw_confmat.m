function draw_confmat(mat,tick) 
%% 
%  ���ߣ�shamoxia.com 
%  ����mat-����tick-Ҫ�����������ʾ��label����������{'label_1','label_2'...} 
% 
%% 
imagesc(mat);            % ���ɫͼ 
% colormap(flipud(hot));  % ת�ɻҶ�ͼ����˸�value�ǽ���ɫ�ģ���value�ǽ��׵� 
colormap(flipud(gray));  % ָ����ɫ����˸�value�ǽ���ɫ�ģ���value�ǽ��׵�Jet Hot Cool Gray
num_class=size(mat,1); 

textStrings = num2str(mat(:),'%.3g'); %���ֵ�С���
textStrings = strtrim(cellstr(textStrings)); 
[x,y] = meshgrid(1:num_class); 
hStrings = text(x(:),y(:),textStrings(:), 'HorizontalAlignment','center','Fontname', 'Times New Roman','FontSize',12); 
midValue = mean(get(gca,'CLim')); 
textColors = repmat(mat(:) > midValue,1,3); %   < �� > �����ɫ����˳��
%�ı�test����ɫ���ں�cell����ʾ��ɫ 
set(hStrings,{'Color'},num2cell(textColors,2));  %# Change the text colors 

if(num_class == 12)  % google
    set(gcf,'Position',[100 100 600 500]);%# �޸�ͼ�Ĵ�С[left bottom width height]
    set(gca,'Position',[.15 .15 .85 .85]);%# ����xy����ͼƬ��ռ�ı���[left bottom width height]
elseif(num_class == 21) % ucm
    set(gcf,'Position',[100 100 900 600]);%# �޸�ͼ�Ĵ�С[left bottom width height]
    set(gca,'Position',[.15 .17 .84 .80]);%# ����xy����ͼƬ��ռ�ı���[left bottom width height]
elseif(num_class==8) % wuhan-iko
    set(gcf,'Position',[100 100 500 400]);%# �޸�ͼ�Ĵ�С[left bottom width height]
    set(gca,'Position',[.28 .25 .72 .75]);%# ����xy����ͼƬ��ռ�ı���[left bottom width height
else % 11 hk-wv3
    set(gcf,'Position',[100 100 500 400]);%# �޸�ͼ�Ĵ�С[left bottom width height]
    set(gca,'Position',[.17 .17 .83 .81]);%# ����xy����ͼƬ��ռ�ı���[left bottom width height]   
end

tick_step = [1:num_class];
tick_step = reshape(tick_step, num_class, 1);
tickx = tick;
for ic = 1 : num_class,
    classname = tick{ic};
    tickx{ic} = strrep(classname, '_', '\_');
end
set(gca,'xtick',tick_step, 'xticklabel',tickx,'Fontname', 'Times new roman','FontSize',16); %# �ֺ�
% set(gca,'xticklabel',tick,'XAxisLocation','top'); 
rotateXLabels(gca, 45 ); 

set(gca,'ytick',tick_step,'yticklabel',tick,'Fontname', 'Times new roman','FontSize',16);%# �ֺ�

colorbar;




